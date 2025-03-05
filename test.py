import glfw

if not glfw.init():
    raise RuntimeError("GLFW could not be initialized")

window = glfw.create_window(800, 600, "Test", None, None)
if not window:
    raise RuntimeError("Failed to create GLFW window")

print("OpenGL initialized successfully!")
glfw.terminate()
