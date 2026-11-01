This is a digital image processing lab, so you’ll learn it best by following one image through the app, then tracing how each algorithm changes it.

### Suggested order

1. **Get the big picture.** Read the [README](/home/mohammed_bahr/Projects/NewProjects/DIP/README.md), especially “Project layout” and “Algorithm map.” The four lecture notes alongside it provide the theory behind each group of algorithms.
2. **Trace the startup path.** Read [main.py](/home/mohammed_bahr/Projects/NewProjects/DIP/main.py), then [main_window.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/gui/main_window.py) to see how the GUI starts and handles user actions.
3. **Find how algorithms connect to the GUI.** Read [operations.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/operations.py). It registers the operations, their parameters, and the functions that run them.
4. **Learn the shared image helpers.** Read [utils.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/utils.py) for image formats, padding, correlation, and other building blocks used across the algorithms.
5. **Study the algorithms in lecture order:**
   - [enhancement.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/enhancement.py) — grayscale, intensity transforms, and histograms
   - [spatial.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/spatial.py) — smoothing, median filters, edges, and Laplacian
   - [frequency.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/frequency.py) — DFT/FFT and frequency filters
   - [features.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/features.py) — Harris corners and HOG
6. **Use a small example to check your understanding.** Read [selftest.py](/home/mohammed_bahr/Projects/NewProjects/DIP/selftest.py) alongside the relevant algorithm. It contains worked examples and expected results. Then look at [demo.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/demo.py) to understand the synthetic test image.
7. **Explore the remaining GUI pieces.** Read [widgets.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/gui/widgets.py) for the image display and plots, then [style.py](/home/mohammed_bahr/Projects/NewProjects/DIP/diplab/gui/style.py) for styling. The GUI is a useful reference after you understand how an operation is registered and implemented.

### For each algorithm

Try tracing this sequence:

`GUI selection → operation and parameters in operations.py → algorithm in its module → helper in utils.py → expected behavior in selftest.py`

Choose one simple operation, such as grayscale conversion or the negative transform, and follow it through that sequence before moving on to the more involved filters.
