import os
import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options

# Paths
SCREENSHOTS_DIR = r"C:\Users\battu\.gemini\antigravity\scratch\agentic-ai-journey\day-10\screenshots"
OUT_DIR = r"C:\Users\battu\.gemini\antigravity\brain\1877360c-242b-4262-b972-86aa59cbeeb1"

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Screenshot configuration
screenshot_files = [
    ('snap1_branch_commit.html', 'day10_snap1_branch_commit.png', 860, 960),
    ('snap2_ast_lint.html', 'day10_snap2_ast_lint.png', 860, 980),
    ('snap3_ci_sim.html', 'day10_snap3_ci_sim.png', 860, 940),
]

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--force-device-scale-factor=1.5')

print("Starting Edge driver...")
driver = webdriver.Edge(options=options)

for html_file, png_name, w, h in screenshot_files:
    url = f'file:///{SCREENSHOTS_DIR}/{html_file}'.replace('\\\\', '/').replace('\\', '/')
    out_path = os.path.join(OUT_DIR, png_name)
    driver.set_window_size(w, h)
    driver.get(url)
    time.sleep(2)  # Allow fonts and styles to render
    driver.save_screenshot(out_path)
    size = os.path.getsize(out_path)
    print(f'Captured: {png_name} ({size:,} bytes) saved to {out_path}')

driver.quit()
print('All screenshots captured successfully!')
