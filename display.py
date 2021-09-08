import sdl2
import sdl2.ext


w = 1164//2
h = 874//2

sdl2.ext.init()


class Display(object):
    def __init__(self,w,h):

        sdl2.ext.init()
        self.w = w
        self.h = h
        self.window = sdl2.ext.Window("calib", size=(self.w, self.h), position=(100,100))
        self.window.show()

    def paint(self,img):

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:,:,0:3] = img.swapaxes(0,1)
        self.window.refresh()
