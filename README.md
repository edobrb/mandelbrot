# Mandelbrot Set Visualizer

![Preview](screen.png)
[Live Demo](https://edobrb.github.io/mandelbrot/)

[Live Demo (the point in this README)](https://edobrb.github.io/mandelbrot/#eyJ4IjoiLTAuMTk5MTAyMTEyNDQ3OTY0NjQ1MDM5ODE3MDY1NDA4NTE4Nzk5MTE4MyIsInkiOiItMS4xMDAxMTg5MDQ3MDk2NzM2NjQ3MzAzODYxMjcwNDg3OTM5OTQwMDYiLCJ2IjoyLjg4NTk1OTcxMDEwMjU5MjNlLTksIm1pIjo3MDAsIm1tIjoiRHluYW1pYyIsImNwIjozMzAsImMiOltbMTcsMTIsOF0sWzUsMjU0LDEwN10sWzk5LDUwLDI0M10sWzI1MSwxNjQsNTRdLFsxOSwyNDgsNzJdLFsyMCwxNywyNF1dLCJ3IjpbMS4wMjQ3OTE2ODA1NTU1MDQ0LDAuOTc1MjA4MzE5NDQyODg2OCwwLjk5OTk5OTk5OTk5OTUwODIsMS4wMDAwMDAwMDAwMDMzODUsMC45OTk5OTk5OTk5OTg3MTY2XSwienMiOjAuODUsInBzIjowLjAxLCJreiI6MC45MiwibWYiOjEuNX0)



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

