from pydantic import BaseModel, validator

class ImageDetails(BaseModel):
    width: int
    height: int
    epsilon: float
    ring_center_width: int
    ring_center_height: int
    min_brightness: int
    max_brightness: int
    
    @validator('width')
    def _check_width(cls, width) -> None:
        if width <= 0:
            raise ValueError(f"Width must be positive integer, not {width}")
        return width
    
    @validator('height')
    def _check_height(cls, height) -> None:
        if height <= 0:
            raise ValueError(f"Height must be positive integer, not {height}")
        return height
        
    @validator('ring_center_width')    
    def _check_width_center(cls, width_center, values, **kwargs) -> None:
        perc_center_diff = 0.05
        width = values['width']
        max_pixel_shift = width * perc_center_diff
        min_width_center = int(width/2 - max_pixel_shift)
        max_width_center = int(width/2 + max_pixel_shift)
        if width_center <= 0:
            raise ValueError(f"Width center must be positive integer, not {width_center}")
        if width_center > width:
            raise ValueError("Can not be greater than width")
        if width_center < min_width_center or width_center > max_width_center:
            raise ValueError(f"Width of the image center must be in range <{min_width_center};{max_width_center}> (up to 5% shift from the default center)")
        return width_center
        
    @validator('ring_center_height')    
    def _check_height_center(cls, height_center, values, **kwargs) -> None:
        perc_center_diff = 0.05
        height = values['height']
        max_pixel_shift = height * perc_center_diff
        min_height_center = int(height/2 - max_pixel_shift)
        max_height_center = int(height/2 + max_pixel_shift)
        if height_center <= 0:
            raise ValueError(f"Height center must be positive integer, not {height_center}")
        if height_center > height:
            raise ValueError("Can not be greater than height")
        if height_center < min_height_center or height_center > max_height_center:
            raise ValueError(f"Height of the image center must be in range <{min_height_center};{max_height_center}> (up to 5% shift from the default center)")
        return height_center

    @validator('min_brightness')
    def _check_min_brightness(cls, min_brightness) -> None:
        if min_brightness < 40 or min_brightness > 120:
            raise ValueError("Minimal brightness must be in in range (40; 120)")
        return min_brightness

    @validator('max_brightness')
    def _check_max_brightness(cls, max_brightness) -> None:
        if max_brightness < 170 or max_brightness > 210:
            raise ValueError("Maximal brightness must be in range (170; 210)")
        return max_brightness
        
    @validator('max_brightness')    
    def _check_min_max(cls, max_brightness, values, **kwargs) -> None:
        min_brightness = values['min_brightness']
        if min_brightness >= max_brightness:
            raise ValueError("Minimal brightness must be smaller than maximal brightness")
        return max_brightness

    @validator('epsilon')
    def _check_epsilon(cls, epsilon) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be in range <0.0; 1.0>")
        return epsilon
    

class ImageFileDetails(ImageDetails):
    filename: str
    
    @validator('filename')
    def _check_filename(cls, filename):
        if filename[-4:] != ".png":
            raise ValueError("Incorrect filename. Please provide filename with extension .png, e.g. \"image.png\".")
    
    
    
if __name__ == "__main__":
    pass