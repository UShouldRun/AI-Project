import pygame

class Window:
    def __init__(self, scale: float) -> None:
        pygame.init()
        self.info = pygame.VideoInfo = pygame.display.Info()
        self.resize(scale)
        pygame.display.set_caption("Connect4")

    def resize(scale: float) -> None:
        self.scale = scale
        if scale == 1.00:
            self.window = pygame.display.set_mode(
                (0,0), pygame.FULLSCREEN
            )
        else:
            self.window = pygame.display.set_mode(
                (self.scale * self.info.current_width, self.scale * self.info.current_height)
            )
