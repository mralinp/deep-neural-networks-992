#! /usr/bin/python3

class Ai:

    def __init__(self):
        self.posDict = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (
            1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2)}

    # If there is a way for winning let's win
    def winPossibilityCheck(board):
        pass

    # Prevent losing
    def preventLosing(board):
        pass


def printBoard(board):

    for row in board:
        a, b, c = row[:]
        print("|%s|%s|%s|" % (a, b, c))


# Is the entered possition valid or not.
def isValid(i, j, board):
    if(i < 4 and i > 0 and j < 4 and j > 0):
        return board[i][j] == ' '
    return False

# Checks that the game has been finished or not.


def endGameCheck(board):
    # Row win check
    for row in board:
        if (row[0] == row[1] and row[1] == row[2] and row[0] != ' '):
            return 1 if (row[1] == 'X') else 2
    # Diameter win check
    if(board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != ' '):
        return 1 if (board[0][0] == 'X') else 2
    if(board[2][0] == board[1][1] and board[1][1] == board[0][2] and board[2][0] != ' '):
        return 1 if (board[2][0] == 'X') else 2
    # Column win check
    for i in range(3):
        if(board[0][i] == board[1][i] and board[0][i] == board[2][i] and board[0][i] != ' '):
            return 1 if (board[0][i] == 'X') else 2
    # No one is winner elsewhere
    return 0


if __name__ == "__main__":
    names = ["computer", "player"]
    board = [[" ", " ", " "] for i in range(3)]
    winner = 0
    ai = Ai()
    for turn in range(9):
        winner = endGameCheck()
        if(winner != 0):
            break
        i, j = ai(board)
        winner = endGameCheck()
        if(winner != 0):
            break
        board[i][j] = 'X'
        printBoard(board)
        userInput = input(
            "Your turn, please enter the cell possition (i.e 1 2): ")
        i, j = list(map(lambda x: int(x) - 1, userInput.split()))
        while(isValid(i, j, board) != True):
            userInput = input(
                "invalid input, please enter valid the cell possition (i.e 1 2): ")
            i, j = list(map(lambda x: int(x) - 1, userInput.split()))
        board[i][j] = 'O'
    if(winner == 0):
        print("tied")
    else:
        print("The winner is %s" % (names[winner-1]))
