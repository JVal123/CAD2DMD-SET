import bpy
import bmesh
import os
import json
import time
import shutil

# ---------- Paths & JSON I/O ----------
def get_blend_key():
    fp = bpy.data.filepath
    if not fp:
        return "unsaved_blend"
    return os.path.splitext(os.path.basename(fp))[0] or "unsaved_blend"

def get_json_path():
    fp = bpy.data.filepath
    base_dir = os.path.dirname(fp) if fp else os.path.expanduser("~")
    return os.path.join(base_dir, "vertex_coords.json")

def load_json_safe(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Keep the invalid file as a backup so we never destroy data
        try:
            backup = f"{path}.invalid.{int(time.time())}.bak"
            shutil.copy2(path, backup)
        except Exception:
            pass
        return {}

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def save_coords_for_current_blend(coords_list):
    """
    coords_list: [[x,y,z]] in the exact user selection order (length 4).
    Writes to { "<blend_name>": [[...]*4], ... } without touching other keys.
    """
    path = get_json_path()
    key = get_blend_key()
    data = load_json_safe(path)
    if not isinstance(data, dict):
        data = {}
    data[key] = coords_list
    atomic_write_json(path, data)
    return path, key

# ---------- Core selection (use user order) ----------
def get_selected_local_coords_in_user_order(obj):
    """
    Returns a list of 4 local coords as lists of floats, in the *user's selection order*.
    Requires selecting exactly 4 vertices one-by-one (Shift+click).
    If Blender can't determine the order, returns an error instead of guessing.
    """
    if obj is None or obj.type != 'MESH':
        return None, "Active object must be a Mesh."
    if obj.mode != 'EDIT':
        return None, "Please be in Edit Mode with exactly four vertices selected."

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Exactly four selected?
    sel_verts = [v for v in bm.verts if v.select]
    if len(sel_verts) != 4:
        return None, f"Select exactly 4 vertices (found {len(sel_verts)})."

    # Build order from selection history (no sorting!)
    # History is oldest -> newest; we keep only currently-selected verts, de-duplicated.
    ordered_history = []
    seen = set()
    for elem in bm.select_history:
        # Only keep verts that are *currently selected*
        if isinstance(elem, bmesh.types.BMVert) and elem.select:
            if elem not in seen:
                ordered_history.append(elem)
                seen.add(elem)

    if len(ordered_history) != 4:
        return None, ("Couldn't determine selection order. "
                      "Please select the 4 vertices one-by-one with Shift+click "
                      "(no box/lasso), then run again.")

    coords = [[float(v.co.x), float(v.co.y), float(v.co.z)] for v in ordered_history]
    return coords, None

# ---------- Operator & UI ----------
class MESH_OT_save_local_coords_in_user_order(bpy.types.Operator):
    """Save 4 LOCAL coords (exact selection order) to JSON under the current .blend name"""
    bl_idname = "mesh.save_local_coords_in_user_order"
    bl_label = "Save 4 Local Coords (Use Selection Order)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        coords, err = get_selected_local_coords_in_user_order(obj)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        # Print & copy
        tags = ["1", "2", "3", "4"]
        pretty_lines = [
            f"{tag}: ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})"
            for tag, c in zip(tags, coords)
        ]
        print("\n=== 4 Vertex LOCAL Coords (user selection order) ===")
        print("\n".join(pretty_lines))

        clip = "[" + ", ".join(f"[{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]" for c in coords) + "]"
        context.window_manager.clipboard = clip

        # Save JSON
        try:
            json_path, blend_key = save_coords_for_current_blend(coords)
            save_msg = f"Saved under key '{blend_key}' to {json_path}"
        except Exception as e:
            save_msg = f"Failed to save JSON: {e}"

        # Warn if unsaved .blend (key will be 'unsaved_blend')
        if blend_key == "unsaved_blend":
            save_msg += "  (Tip: Save your .blend to use its filename as the JSON key.)"

        # Popup
        def draw(self_popup, _ctx):
            for line in pretty_lines:
                self_popup.layout.label(text=line)
            self_popup.layout.separator()
            self_popup.layout.label(text="Order used: exactly as you selected")
            self_popup.layout.label(text=save_msg)

        context.window_manager.popup_menu(
            draw, title="4 Local Coords (copied & saved)", icon='INFO'
        )

        self.report({'INFO'}, "Saved 4 local coords (selection order).")
        return {'FINISHED'}

class VIEW3D_PT_coords_saver(bpy.types.Panel):
    bl_label = "Local Coords"
    bl_idname = "VIEW3D_PT_coords_saver"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Coords'

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        obj = context.active_object
        sel_count = None
        if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            sel_count = sum(1 for v in bm.verts if v.select)
        col.label(text=f"Selected verts: {sel_count}" if sel_count is not None else "Select a mesh in Edit Mode")
        col.operator("mesh.save_local_coords_in_user_order", icon='OUTLINER_OB_MESH')

def menu_func(self, context):
    self.layout.operator(MESH_OT_save_local_coords_in_user_order.bl_idname,
                         text="Save 4 Local Coords (Use Selection Order)",
                         icon='OUTLINER_OB_MESH')

classes = (
    MESH_OT_save_local_coords_in_user_order,
    VIEW3D_PT_coords_saver,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_edit_mesh.append(menu_func)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(menu_func)

def unregister():
    bpy.types.VIEW3D_MT_edit_mesh.remove(menu_func)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
