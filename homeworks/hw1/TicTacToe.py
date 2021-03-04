class TicTacToe:

    def __init__(self):
        self.posDict = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (
            1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2)}
        self.board = [[" ", " ", " "] for i in range(3)]

    def place(self, i, j, c=' '):
        if(self.isValid(i, j)):
            self.board[i][j] = c
            return False
        return True

    # Is the entered possition valid or not.
    def isValid(self, i, j):
        if(i < 4 and i >= 0 and j < 4 and j >= 0):
            return self.board[i][j] == ' '
        return False

    def endGameCheck(self):
        # Row win check
        for row in self.board:
            if (row[0] == row[1] and row[1] == row[2] and row[0] != ' '):
                return 1 if (row[1] == 'X') else 2
        # Diameter win check
        if(self.board[0][0] == self.board[1][1] and self.board[1][1] == self.board[2][2] and self.board[0][0] != ' '):
            return 1 if (self.board[0][0] == 'X') else 2
        if(self.board[2][0] == self.board[1][1] and self.board[1][1] == self.board[0][2] and self.board[2][0] != ' '):
            return 1 if (self.board[2][0] == 'X') else 2
        # Column win check
        for i in range(3):
            if(self.board[0][i] == self.board[1][i] and self.board[0][i] == self.board[2][i] and self.board[0][i] != ' '):
                return 1 if (self.board[0][i] == 'X') else 2
        # No one is winner elsewhere
        return 0

    # If there is a way for winning, let's win
    def __winPossibilityCheck(self):
        for i in range(3):
            for j in range(3):
                if self.isValid(i, j):
                    self.board[i][j] = 'X'
                    if(self.endGameCheck() == 1):
                        self.board[i][j] = ' '
                        return (i, j)
                    self.board[i][j] = ' '

        return (-1, -1)

    # Prevent losing
    def __preventLosing(self):
        for i in range(3):
            for j in range(3):
                if self.isValid(i, j):
                    self.board[i][j] = 'O'
                    if(self.endGameCheck() == 2):
                        self.board[i][j] = ' '
                        return (i, j)
                    self.board[i][j] = ' '

        return (-1, -1)

    # if turn is 0 mark center.
    def __firstTrunRule(self, turn):
        if(turn == 1):
            return (1, 1)
        return (-1, -1)

    # If you are at thred turn.
    def __theredTurn(self, turn):
        conditions = [
            [
                [' ', 'O', ' '],
                [' ', 'X', ' '],
                [' ', ' ', ' ']
            ],
            [
                [' ', ' ', ' '],
                [' ', 'X', 'O'],
                [' ', ' ', ' ']
            ],
            [
                [' ', ' ', ' '],
                [' ', 'X', ' '],
                [' ', 'O', ' ']
            ],
            [
                [' ', ' ', ' '],
                ['O', 'X', ' '],
                [' ', ' ', ' ']
            ]
        ]
        d = [(1, 2), (0, 1), (1, 2), (0, 1)]
        if(turn == 3):
            for i in range(4):
                if(self.__equals(conditions[i])):
                    return d[i]
        return (-1, -1)

    def __equals(self, board):
        for i in range(3):
            for j in range(3):
                if(board[i][j] != self.board[i][j]):
                    return False
        return True

    def __fivthTurn(self, turn):
        conditions = [
            [
                [' ', 'O', ' '],
                ['O', 'X', 'X'],
                [' ', ' ', ' ']
            ],
            [
                [' ', 'X', ' '],
                [' ', 'X', 'O'],
                [' ', 'O', ' ']
            ],
            [
                [' ', 'O', ' '],
                [' ', 'X', 'X'],
                [' ', 'O', ' ']
            ],
            [
                [' ', ' ', ' '],
                ['O', 'X', 'X'],
                [' ', 'O', ' ']
            ]
        ]
        d = [(0, 2), (0, 2), (2, 2), (2, 2)]

        if(turn == 5):
            print("hit5")
            for i in range(4):
                if(self.__equals(conditions[i])):
                    return d[i]
        return (-1, -1)

    def takeTurn(self, turn):
        p = self.__winPossibilityCheck()
        if(p != (-1, -1)):
            return p
        p = self.__preventLosing()
        if(p != (-1, -1)):
            return p
        p = self.__firstTrunRule(turn)
        if(p != (-1, -1)):
            return p
        p = self.__theredTurn(turn)
        if(p != (-1, -1)):
            return p
        p = self.__fivthTurn(turn)
        if(p != (-1, -1)):
            return p
        for i in range(3):
            for j in range(3):
                if(self.isValid(i, j)):
                    return (i, j)
        return (-1, -1)

    def printBoard(self):
        for row in self.board:
            a, b, c = row[:]
            print("|%s|%s|%s|" % (a, b, c))


if __name__ == "__main__":
    names = ["computer", "player"]
    ai = TicTacToe()
    winner = 0
    for turn in range(9):
        if(turn % 2 == 0):
            i, j = ai.takeTurn(turn+1)
            ai.place(i, j, 'X')
        else:
            userInput = input(
                "Your turn, please enter the cell possition (i.e 1 2): ")
            i, j = list(map(lambda x: int(x) - 1, userInput.split()))
            while(ai.isValid(i, j) != True):
                userInput = input(
                    "invalid input, please enter valid the cell possition (i.e 1 2): ")
                i, j = list(map(lambda x: int(x) - 1, userInput.split()))
            ai.place(i, j, 'O')
        winner = ai.endGameCheck()
        ai.printBoard()
        print("-------------")
        if(winner != 0):
            break

    if(winner == 0):
        print("tied")
    else:
        print("The winner is %s" % (names[winner-1]))
