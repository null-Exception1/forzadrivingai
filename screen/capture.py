import pyautogui

class Capture:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    # snap a frame
    def frame(self):
        im = pyautogui.screenshot(region=(self.x,self.y, self.w, self.h))
        return im

    

