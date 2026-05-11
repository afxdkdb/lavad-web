import sys
sys.path.insert(0, "/home/jinanyang/lavad/libs/ImageBind")

try:
    from imagebind.models.imagebind_model import imagebind_huge
    print("Import successful!")
    print("Loading ImageBind model...")
    model = imagebind_huge(pretrained=True)
    print("ImageBind loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()