# Mandelbrot Set Visualizer

![Preview](screen.png)
[Live Demo](https://edobrb.github.io/mandelbrot/)

[Live Demo (the point in this README)](https://edobrb.github.io/mandelbrot/#eyJ4IjoiLTAuMTk5MTAyMTEyMjI4MDY3NzE0OTQ1NzIxNDIyNjcwOTUyMDIxODEzIiwieSI6Ii0xLjEwMDExODkwNDY4MDgxNDA2NzYyOTM2MDIwMjI4MTk4NzY5NzEyOTciLCJ2IjoyLjg4NTk1OTcxMDEwMjU5MjNlLTksIm1pIjoyOTI5LCJtbSI6IkR5bmFtaWMiLCJjcCI6NDI2LCJjIjpbWzE3LDEyLDhdLFs1LDI1NCwxMDddLFs5OSw1MCwyNDNdLFsyNTEsMTY0LDU0XSxbMTksMjQ4LDcyXSxbMjAsMTcsMjRdXSwidyI6WzEsMSwxLDEsMV0sInpzIjowLjg1LCJwcyI6MC4wMSwia3oiOjAuOTIsIm1mIjoxLjV9)



Real-time Mandelbrot set explorer running in the browser using **WebGPU** and perturbation theory for deep zoom with arbitrary precision.

## Requirements

A browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly).

## Controls

### Desktop
| Input | Action |
|---|---|
| Left drag | Pan |
| Scroll wheel | Zoom |
| W / S | Zoom in / out |
| Arrow keys | Pan |
| I | Toggle settings panel |
| F11 | Toggle fullscreen |
| F12 | Screenshot |

### Smartphone / Touch
| Input | Action |
|---|---|
| Drag | Pan |
| Pinch | Zoom |
| ☰ button | Toggle settings panel |

Enable **Smartphone mode** in the Navigation section of the settings panel to show the floating ☰ toggle button.

## Settings Panel

Open with `I` (desktop) or the ☰ button (smartphone mode).

| Section | Setting | Description |
|---|---|---|
| Rendering | Max iter mode | `Dynamic` (auto-scales with zoom) or `Fixed` |
| Rendering | Max iterations | Base iteration limit |
| Colors | Color period | Iterations before the palette cycles |
| Colors | Color stops | Gradient control points |
| Colors | Segment weights | Relative size of gradient segments |
| Navigation | Scroll zoom speed | Zoom factor per scroll tick |
| Navigation | Key zoom speed | Zoom factor per W/S keypress |
| Navigation | Pan speed | Fraction of viewport moved per arrow key |
| Navigation | Smartphone mode | Shows floating ☰ button; switches help text to touch instructions |
| Bookmarks | — | Save and recall named locations |

Settings are persisted to `localStorage` via **Save settings**.

