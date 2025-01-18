import tkinter as tk
from tkinter import ttk, colorchooser
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import numpy as np

from src.vision import *
from src.vision import _match_color_hue

def match_random(image):
    "Match random pixels of the image"
    return np.random.randint(0, 2, size=(image.shape[0], image.shape[1]))

def match_black(image):
    "Matches all black pixels in the source image"
    return np.all(image == [0, 0, 0], axis=-1).astype(int)

def match_by_pixel_value(image, threshold: int):
    "Match pixels where the sum of RGB channels is greater than or equal to the threshold"
    return (np.sum(image, axis=-1) >= threshold).astype(int)

def match_color(image, match_color, tolerance: int=0):
    "Match all pixels having the specified color. Non-zero match tolerance may optionally be set to match more colors"
    color_diff = np.abs(image - match_color).sum(axis=-1)
    return (color_diff <= tolerance).astype(int)

def inverted_color_match(image, exclude_color, tolerance: int=0):
    "Match all colors except the specified one. Non-zero tolerance may optionally be set to match less colors"
    color_diff = np.abs(image - exclude_color).sum(axis=-1)
    return (color_diff > tolerance).astype(int)

class MatchFunctionEntry:
    def __init__(self, func, name, arg_restrictions=[]):
        self.func = func
        self.name = name
        self.arg_restrictions = arg_restrictions

class Slider:
    def __init__(self, parent, label_text, range_, row, update_callback, is_integer):
        self.frame = ttk.Frame(parent)
        self.frame.grid(row=row, column=0, pady=5, sticky="ew")

        self.var = tk.IntVar() if is_integer else tk.DoubleVar()
        self.label = ttk.Label(self.frame, text=f"{label_text}:")
        self.label.grid(row=0, column=0, padx=5, sticky="ew")

        self.entry = ttk.Entry(self.frame, textvariable=self.var, width=5)
        self.entry.grid(row=0, column=1, padx=5, sticky="ew")
        self.entry.bind('<Return>', lambda event: update_callback())

        self.slider = ttk.Scale(self.frame, from_=range_[0], to=range_[1], orient=tk.HORIZONTAL, variable=self.var, command=lambda value: update_callback())
        self.slider.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        # Ensure the slider stretches with the frame
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

    def get_value(self):
        return self.var.get()

class ColorPickerButton:
    def __init__(self, parent, label_text, update_callback, initial_color='#000000'):
        self.frame = ttk.Frame(parent)

        self.var = tk.StringVar(value=initial_color)
        self.label = ttk.Label(self.frame, text=f"{label_text}:")
        self.label.pack(side=tk.LEFT, padx=5)

        self.button = ttk.Button(self.frame, command=self.select_color, width=5)
        self.button.pack(side=tk.LEFT, padx=5)

        self.entry = ttk.Entry(self.frame, textvariable=self.var, width=10)
        self.entry.pack(side=tk.LEFT, padx=5)
        self.entry.bind('<Return>', lambda event: self.update_color_from_entry())

        # Trace the variable to update the button color
        self.var.trace_add("write", lambda *args: self.update_color_button())
        self.update_callback = update_callback

        # Set initial color
        self.update_color_button()

    def select_color(self):
        color = colorchooser.askcolor(title="Choose color")[1]
        if color:
            self.var.set(color)
            self.update_callback()

    def update_color_button(self):
        style_name = f"Color.{id(self.button)}.TButton"
        style = ttk.Style()
        style.theme_use('alt')
        if style_name not in style.map("TButton"):
            style.map(style_name, background=[("active", self.var.get()), ("!active", self.var.get())])
        self.button.configure(style=style_name)

    def update_color_from_entry(self):
        color_hex = self.var.get().lstrip('#')
        if len(color_hex) == 6 and all(c in '0123456789abcdefABCDEF' for c in color_hex):
            color_hex = '#' + color_hex
            self.var.set(color_hex)
            self.update_color_button()
            self.update_callback()

    def get_value(self):
        color_hex = self.var.get()
        color_rgb = np.array([int(color_hex[i:i+2], 16) for i in (1, 3, 5)], dtype=np.uint8)
        return color_rgb

class PixelMatcherApp:
    def __init__(self, root, image_path, modifier_image_path=None):
        self.root = root
        self.root.title("Welcome to Pixel Matcher!")

        # Create a frame to hold the plot
        self.frame = ttk.Frame(root)
        self.frame.grid(row=0, column=0, sticky="nsew")

        # Create a figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create a canvas to display the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Load the initial image and update the display
        self.set_source_image(mpimg.imread(image_path))

        # Load and process the modifier image
        self.set_modifier_image(modifier_image_path)

        # Create a frame for interactive elements
        self.interactive_frame = ttk.Frame(root)
        self.interactive_frame.grid(row=0, column=1, sticky="nsew")
        self.next_interactive_row = 0

        # Initialize the list of match functions
        self.match_functions = [
            MatchFunctionEntry(match_random, "Random match"),
            MatchFunctionEntry(match_black, "Match black"),
            MatchFunctionEntry(match_by_pixel_value, "Match by value", [('threshold', 'int_slider', (0, 765))]),
            MatchFunctionEntry(match_color, "Color Match", [('match_color', 'color_picker', None), ('tolerance', 'int_slider', (0, 765))]),
            MatchFunctionEntry(inverted_color_match, "Color Exclude", [('exclude_color', 'color_picker', None), ('tolerance', 'int_slider', (0, 765))]),
            MatchFunctionEntry(match_color_grades, "Match tone", [('color', 'color_picker', None), ('min_pixel_value', 'int_slider', (0, 765)), ('tolerance', 'int_slider', (0, 765))]),
            MatchFunctionEntry(match_color_grades_std, "Match tone std", [('color', 'color_picker', None), ('min_pixel_value', 'int_slider', (0, 765)), ('tolerance', 'float_slider', (0, 100))]),
            MatchFunctionEntry(match_color_hue, "Match hue", [('color', 'color_picker', None), ('min_pixel_value', 'int_slider', (0, 765)), ('tolerance', 'int_slider', (0, 180))]),
            MatchFunctionEntry(_match_color_hue, "Match hue direct", [('hue', 'int_slider', (0, 180)), ('min_pixel_value', 'int_slider', (0, 765)), ('tolerance', 'int_slider', (0, 180))])
        ]

        self.collapse_button = ttk.Button(self.interactive_frame, text="Collapse panel", command=self.toggle_interactive_frame)
        self.add_interactive_widget(self.collapse_button)

        # Create a dropdown for match function selection
        self.match_selector_var = tk.StringVar()
        self.match_selector_var.set(self.match_functions[0].name)
        self.match_selector = ttk.OptionMenu(self.interactive_frame, self.match_selector_var, self.match_selector_var.get(), *[entry.name for entry in self.match_functions], command=self.update_match_ui)
        self.add_interactive_widget(self.match_selector)

        # Create a button to update the image
        self.update_button = ttk.Button(self.interactive_frame, text="Make random image", command=self.generate_and_update_image)
        self.add_interactive_widget(self.update_button)

        # Create a checkbox to toggle match display
        self.display_match_checkbox_var = tk.BooleanVar(value=True)
        self.display_match_checkbox = ttk.Checkbutton(self.interactive_frame, text="Show Match", variable=self.display_match_checkbox_var, command=self.toggle_match)
        self.add_interactive_widget(self.display_match_checkbox)

        # Create a checkbox to control match display mode
        self.overlay_match_checkbox_var = tk.BooleanVar(value=True)
        self.overlay_match_checkbox = ttk.Checkbutton(self.interactive_frame, text="Overlay Match", variable=self.overlay_match_checkbox_var, command=self.update_match)
        self.add_interactive_widget(self.overlay_match_checkbox)

        # Add a color picker button to select match pixels color
        self.match_color_picker = ColorPickerButton(parent=self.interactive_frame, label_text="Select Match Color", update_callback=self.update_match, initial_color='#FF0000')
        self.add_interactive_widget(self.match_color_picker.frame)

        # Add a label to display the count of matched pixels and percentage
        self.match_info_label = ttk.Label(self.interactive_frame, text="match_info_label")
        self.add_interactive_widget(self.match_info_label)

        self.match_ext_info_label = ttk.Label(self.interactive_frame, text="match_ext_info_label")
        self.add_interactive_widget(self.match_ext_info_label)

        self.best_color_match_label = ttk.Label(self.interactive_frame, text="<color jitter result>")
        self.add_interactive_widget(self.best_color_match_label)

        def open_image():
            plt.imshow(self.displayed_image)
            plt.show()

        self.open_image_button = ttk.Button(self.interactive_frame, text="Open image separately", command=  open_image)
        self.add_interactive_widget(self.open_image_button)

        # Add a separator
        self.separator = ttk.Separator(self.interactive_frame, orient=tk.HORIZONTAL)
        self.add_interactive_widget(self.separator)

        def on_jitter_button_click():
            if not self.jitter_in_progress:
                self.jitter_in_progress = True
                self.max_match_score = 0
                self.temperature = 750

                # set the first color argument in the list
                self.jitter_target_widget = None
                for widget in self.additional_ui_elements:
                    if isinstance(widget, ColorPickerButton):
                        self.jitter_target_widget = widget
                        break
                    
                if self.jitter_target_widget is None:
                    print(f'Error: no color argument found')
                    return

                self.root.after(20, self.do_jitter_color)
            else:
                self.jitter_in_progress = False

        self.jitter_button = ttk.Button(self.interactive_frame, text="Jitter color", command=on_jitter_button_click)
        self.add_interactive_widget(self.jitter_button)

        # Adjust the layout
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.interactive_frame.grid_columnconfigure(0, weight=1)
        self.interactive_frame.grid_rowconfigure(self.next_interactive_row, weight=1)  # Ensure the row with additional UI elements expands

        # Initialize the additional UI elements
        self.additional_ui_elements = []
        self.maxed_color = (np.zeros(3, dtype=np.uint8), '#000000')
        self.jitter_in_progress = False
        self.interactive_collapsed = False

        # Calculate match
        self.update_match()

    def toggle_interactive_frame(self):
        if self.interactive_collapsed:
            self.root.grid_columnconfigure(0, weight=2)
        else:
            self.root.grid_columnconfigure(0, weight=25)

        self.interactive_collapsed = not self.interactive_collapsed

    def add_interactive_widget(self, widget):
        widget.grid(row=self.next_interactive_row, column=0, pady=10, sticky="ew")
        self.next_interactive_row += 1

    def do_jitter_color(self):
        if not self.jitter_in_progress: return
        random_color = ((self.maxed_color[0] + self.temperature * (np.random.rand(3) - 0.5)) % 256).astype(np.uint8)
        color_hex = '#' + format(random_color[0], '02x') + format(random_color[1], '02x') + format(random_color[2], '02x')

        # set the first color argument in the list
        self.jitter_target_widget.var.set(color_hex)
        
        self.update_match()

        self.max_match_score = max(self.max_match_score, self.modified_match_score)
        if self.max_match_score == self.modified_match_score:
            self.maxed_color = (random_color, color_hex)
        
        self.best_color_match_label.config(text=f'{self.maxed_color[1]}: {self.max_match_score} ({self.temperature:0.1f} K)')
        print(f'New color: {color_hex}, match score: {self.modified_match_score}, max reached: {self.max_match_score} ({self.maxed_color[1]}), T: {self.temperature:0.1f} K')

        if self.temperature > 10:
            self.temperature *= 0.999
        else:
            self.temperature *= 0.997

        if self.temperature > 0.5:
            self.root.after(5, self.do_jitter_color)
        else:
            print('Temperature too low, stopping the jitter')
            self.jitter_in_progress = False

    def set_source_image(self, new_image):
        self.source_image = (new_image * 255).astype(np.uint8)
        self.update_image(self.source_image)

    def set_modifier_image(self, modifier_image_path):
        if modifier_image_path:
            modifier_image = mpimg.imread(modifier_image_path)
            if modifier_image.shape[:2] != self.source_image.shape[:2]:
                raise ValueError("Modifier image dimensions do not match source image dimensions.")
            red_channel = modifier_image[:, :, 0] # negative channel, 255 == -1, 254 == -2, ..., 0 == 0
            green_channel = modifier_image[:, :, 1] # positive channel, 255 == 1, 254 == 2, ..., 0 == 0
            self.modifier_image = ((256 - 255 * green_channel) % 256 - (256 - 255 * red_channel) % 256).astype(int)
        else:
            self.modifier_image = np.ones_like(self.source_image[:, :, 0], dtype=int)

    def update_image(self, new_image):
        self.ax.clear()  # Clear the current plot
        self.ax.imshow(new_image)
        self.ax.axis('off')  # Hide the axes
        self.canvas.draw()  # Redraw the canvas

    def generate_and_update_image(self):
        new_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Generate a random image
        self.set_source_image(new_image)

    def toggle_match(self):
        if self.display_match_checkbox_var.get():
            self.update_match()
        else:
            self.update_image(self.source_image)

    def update_match_ui(self, *args):
        # Clear existing additional UI elements
        for element in self.additional_ui_elements:
            element.frame.grid_forget()
        self.additional_ui_elements.clear()

        # Get the selected match function
        selected_match_function = next(entry for entry in self.match_functions if entry.name == self.match_selector_var.get())

        # Create additional UI elements based on argument restrictions
        row = self.next_interactive_row
        for arg_name, arg_type, arg_range in selected_match_function.arg_restrictions:
            if arg_type == 'int_slider':
                element = Slider(self.interactive_frame, arg_name, arg_range, row, self.update_match, is_integer=True)
                self.additional_ui_elements.append(element)
                row += 1  # Move to the next set of rows
            elif arg_type == 'float_slider':
                element = Slider(self.interactive_frame, arg_name, arg_range, row, self.update_match, is_integer=False)
                self.additional_ui_elements.append(element)
                row += 1  # Move to the next set of rows
            elif arg_type == 'color_picker':
                element = ColorPickerButton(self.interactive_frame, arg_name, self.update_match)
                element.frame.grid(row=row, column=0, pady=10, sticky="ew")
                self.additional_ui_elements.append(element)
                row += 1  # Move to the next set of rows

        # Regenerate the match when a new match function is selected
        self.update_match()

    def update_match(self):
        selected_match_function = next(entry for entry in self.match_functions if entry.name == self.match_selector_var.get())
        args = []
        for element in self.additional_ui_elements:
            args.append(element.get_value())
        self.current_match = selected_match_function.func(self.source_image, *args)

        # Update the match info label
        total_pixels = self.current_match.size
        self.matched_pixels_number = np.sum(self.current_match)
        matched_percentage = (self.matched_pixels_number / total_pixels) * 100
        self.match_info_label.config(text=f"Matched Pixels: {self.matched_pixels_number} ({matched_percentage:.2f}%)")

        # compute extended match data
        self.modified_match_score = np.sum(self.modifier_image[self.current_match.astype(bool)])
        self.match_ext_info_label.config(text=f"Modified match score: {self.modified_match_score}")

        if self.display_match_checkbox_var.get():
            # Determine the base image for overlay
            base_image = self.source_image
            if not self.overlay_match_checkbox_var.get():
                base_image = np.zeros_like(self.source_image)

            # Overlay the match on the base image
            match_color = self.match_color_picker.get_value()
            match_rgb = match_color / 255.0
            self.displayed_image = np.where(self.current_match[..., np.newaxis] == 1, match_rgb, base_image / 255.0)
            self.update_image(self.displayed_image)

    def update_color_button(self, button, color):
        style_name = f"Color.{id(button)}.TButton"
        style = ttk.Style()
        style.theme_use('alt')
        if style_name not in style.map("TButton"):
            style.map(style_name, background=[("active", color), ("!active", color)])
        button.configure(style=style_name)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Display an image on a plot.')
    parser.add_argument('--source-image', type=str, required=True, help='Path to the source image')
    parser.add_argument('--modifier-image', type=str, help='Path to the modifier image')
    args = parser.parse_args()

    # Create the main window
    root = tk.Tk()

    # Create an instance of PixelMatcherApp
    app = PixelMatcherApp(root, args.source_image, args.modifier_image)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
