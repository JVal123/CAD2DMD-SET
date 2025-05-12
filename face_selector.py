import bpy
import json
import os


class ExportSelectedFace(bpy.types.Operator):
    """Wait until one face is selected, and then export its index into a json file"""
    bl_idname = "object.export_selected_face"
    bl_label = "Export Selected Face Index"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}
        
        if event.type == 'TIMER':
            obj = context.active_object
            if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
                # Force update by switching modes
                bpy.ops.object.mode_set(mode='OBJECT')
                selected_faces = [f for f in obj.data.polygons if f.select]
                if len(selected_faces) > 0:
                    selected_face_index = selected_faces[0].index #We assume it's the first face in case several have been selected

                    # Get the current blend file path and name
                    blend_filepath = bpy.data.filepath
                    blend_filename = os.path.splitext(os.path.basename(blend_filepath))[0]

                    # Construct JSON data
                    export_data = {
                        f"{blend_filename}": selected_face_index
                    }
                    
                    # Save coordinates to a JSON file in the same folder as the blend file.
                    filepath = os.path.join(bpy.path.abspath("//"), "face_indices.json")

                    # Checks if the file exists
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            try:
                                existing_data = json.load(f)

                                # Append the new data
                                existing_data.update(export_data)

                                with open(filepath, 'w') as f:
                                    json.dump(existing_data, f, indent=4)
                            except json.JSONDecodeError:
                                existing_data = []  # Handle empty or corrupted JSON file
                    else:
                        with open(filepath, 'w') as f:
                            json.dump(export_data, f, indent=4)

                    self.report({'INFO'}, f"Exported face index to {filepath}")
                    
                    # Optionally, switch back to Edit Mode:
                    bpy.ops.object.mode_set(mode='EDIT')
                    
                    self.cancel(context)
                    return {'FINISHED'}
                else:
                    bpy.ops.object.mode_set(mode='EDIT')
        return {'PASS_THROUGH'}

    def execute(self, context):
        obj = context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        if not (obj and obj.type == 'MESH'):
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Object must be in Edit Mode")
            return {'CANCELLED'}
        
        wm = context.window_manager # Gets the Window Manager from the current context, which is responsible for handling events and timers
        self._timer = wm.event_timer_add(1, window=context.window) # Adds a timer event that activates every second, and is tied to the current window
        wm.modal_handler_add(self) # Registers the modal operator
        self.report({'INFO'}, "Waiting for a face to be selected...")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
        self.report({'INFO'}, "Operator canceled.")

def register():
    bpy.utils.register_class(ExportSelectedFace)

def unregister():
    bpy.utils.unregister_class(ExportSelectedFace)

if __name__ == "__main__":
    # Register the operator
    register()
    
    # Start the operator for interactive face selection.
    bpy.ops.object.export_selected_face()