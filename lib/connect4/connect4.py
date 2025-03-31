class Connect4:


    def __init__(self, int x, int y):
        self.turn =1 #1 for player 1 and 2 for player 2
        self.nx=x
        self.ny=y
        self.board : [[0]*y for _ in range(x)]
        
        self.dirs = [(0, 1), (1, 0), (1, 1), (-1, 1), (-1, -1), (-1, 0), (0, -1), (1, -1)]

        self.final= False # true if the game has ended
            
    def play(self,int x, int y) -> bool: 
        if self.isValidMove(x,y):
            self.board[x][y]= self.turn
            self.checkWining(x,y)
            self.changeTurn()
            return True
        return False
        
    def isValidMove(self,int x, int y) -> bool:
        if self.is_out_of_bounds(x, y) or self.board[x][y]!=0:
            return False

        if y == 0 or self.board[x][y - 1] != 0:  
            return True

        return False   

    def getTurn(self) -> int: 
        return self.turn

    def changeTurn(self):
        self.turn= 2 if self.turn==1 else 1

    def getValidMoves(self) -> [[int]]:
        moves = []
        for i in range(self.nx):
            for j in range(self.ny):
                if self.isValidMove(i,j):
                    moves.append((i,j))
        return moves

    def isOutOfBounds(self, x: int, y: int) -> bool:
        return x>=self.nx or x<0 or y>=self.ny or y<0

    def clearBoard(self):
        self.board = [[0] * self.ny for _ in range(self.nx)]
    
    def isFinal(self) -> bool: 
        
        return self.final

    def checkWining(self, x: int, y: int) :
        #checks if last move led to a win in all the directions
        for dx, dy in self.dirs:
            count=1 #start with the newly placed piece
            count+=countInDirection(x, y, dx, dy, True)
            count+=countInDirection(x, y, dx, dy, False)
            if(count>=4):
                self.final=True
                return 

        self.final=False
        return

    def countInDirection(self,x: int, y: int, dx: int, dy: int, positive: bool) ->int:
        count=0
        while True:
            x = x + dx if positive else x - dx
            y = y + dy if positive else y - dy

            if self.isOutOfBounds(x, y) or self.board[x][y] != self.turn:
                break

            count += 1

        return count




