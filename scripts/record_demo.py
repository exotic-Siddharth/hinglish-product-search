"""Record the demo page as a GIF (assets/demo.gif): type two Hinglish queries, show the results.

Start the server first (see the README), then:
    pip install playwright      # drives the installed Chrome or Edge; no browser download
    python scripts/record_demo.py assets/demo.gif
"""
import io
import sys

from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000/"
QUERIES = ["gents ke liye kaala jacket", "bachon ke liye pani wali botal"]
W, H = 860, 700
frames = []  # (image, duration_ms)


def snap(page, ms):
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    if frames and frames[-1][0].tobytes() == img.tobytes():
        frames[-1] = (frames[-1][0], frames[-1][1] + ms)
    else:
        frames.append((img, ms))


def main(out):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except PlaywrightError:  # no Chrome installed: use Edge
            browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(URL)
        snap(page, 1200)
        for n, q in enumerate(QUERIES):
            page.fill("input[name=q]", "")
            page.click("input[name=q]")
            if n:
                snap(page, 500)
            for ch in q:
                page.keyboard.type(ch)
                snap(page, 70)
            snap(page, 400)
            with page.expect_navigation():
                page.keyboard.press("Enter")
            page.wait_for_load_state("networkidle")
            snap(page, 3800)
        browser.close()
    imgs = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f, _ in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=[d for _, d in frames], loop=0,
                 optimize=True, disposal=1)
    print(f"{len(frames)} frames, {sum(d for _, d in frames) / 1000:.1f} s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
