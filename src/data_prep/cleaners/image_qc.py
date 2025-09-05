from PIL import Image
def valid_image(path: str, min_w=256, min_h=256):
    try:
        im = Image.open(path); w,h = im.size
        return (w>=min_w and h>=min_h)
    except Exception:
        return False
