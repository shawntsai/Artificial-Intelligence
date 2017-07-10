import sys
import copy
import getopt


#               a   b   c  d  e  f  g    h 
board_value = [[99, -8, 8, 6, 6, 8, -8, 99],    
              [-8, -24, -4, -3, -3, -4, -24, -8],
              [8, -4, 7, 4, 4, 7, -4, 8],
              [6, -3, 4, 0, 0, 4, -3, 6],
              [6, -3, 4, 0, 0, 4, -3, 6],
              [8, -4, 7, 4, 4, 7, -4, 8],
              [-8, -24, -4, -3, -3, -4, -24, -8],
              [99, -8, 8, 6, 6, 8, -8, 99]]    

directions = [[-1, 1], [0, 1], [1, 1],
              [-1, 0],         [1, 0],
              [-1, -1],[0, -1],[1, -1]]

mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

def valid(coor):
    return (coor[0] >= 0 and coor[0] < 8 and coor[1] >= 0 and coor[1] < 8)

def next_tuple(origin, direction):
    return (origin[0] + direction[0], origin[1] + direction[1])

class Board(object):
    def __init__(self, move, depth, board, player):
        self.board = board
        self.player = player 
        self._len = len(board)
        self.parent = None
        self.opponent = 'X' if player == 'O' else 'O'
        self.next = None
        # self.alpha = None
        # self.beta = None
        self.move = move
        self.depth = depth
          
    def board_str(self):
        y = ""
        for i in range(8):
            x = ""
            for value in self.board[i]:
                x += "".join(value)  
            y += x 
            if i != 7:
                y += '\n'
        return y


    def __len__(self):
        return self._len

    def __str__(self):
        y = self.board_str()
        return y

    def write_board(self):
        f = open('output.txt', 'w')
        s = self.board_str()
        f.write(s + '\n')
        f.close()

    def get_valid_moves(self):
        moves = list()
        for i in range(0, 8):
            for j in range(0, 8):
                val = self.board[i][j]
                if val == '*':
                    for direction in directions:
                        move = next_tuple((i, j), direction)
                        while valid(move) and self.board[move[0]][move[1]] == self.opponent:
                            move = next_tuple(move, direction)
                            if valid(move) and self.board[move[0]][move[1]] == self.player:
                                moves.append(((i, j), direction))
        return moves
                            
    def make_move(self, move):
        if move == None:
            return board
        new_board = copy.deepcopy(self.board)
        origin = move[0]
        direction = move[1]
        # print origin[0]
        # print direction

        new_board[origin[0]][origin[1]] = self.player
        flip = next_tuple(origin, direction)
        while new_board[flip[0]][flip[1]] == self.opponent:
            new_board[flip[0]][flip[1]] = self.player
            flip = next_tuple(flip, direction)
        move_name = "{}{}".format(mapping[move[0][1]], move[0][0] + 1)
        return Board(move_name, self.depth + 1, new_board, self.opponent)


    def output_state(self):
        output_file = 'output.txt'
        f = open(output_file, 'w')
        for i in range(0, len(self.board)):
            f.write("".join(self.board[i]) + "\n")
        f.close()
        

    def is_game_over(self):
        x = 0
        o = 0
        n = 0
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == 'O':
                    o += 1
                elif self.board[i][j] == 'X':
                    x += 1
                else:
                    n += 1

        return True if x == 0 or o == 0 or n == 0  else False


class AlphaBetaPruning(object):
    MAXIMUM = sys.maxint
    MINIMUM = -sys.maxint - 1

    def __init__(self, board, depth, player):
        self.board = board
        self.max_depth = depth
        self.traverse_log = list()
        self.traverse_log.append("Node,Depth,Value,Alpha,Beta")
        self.player = player
        self.best_board = board
        self.opponent = 'O' if player == 'X' else 'O'

    def evaluate(self, board):
        player = self.player
        opponent = self.opponent
        evaluation = 0
        for i in range(len(board.board)):
            for j in range(len(board.board)):
                if board.board[i][j] == player:
                    evaluation += board_value[i][j]
                elif board.board[i][j] == opponent:
                    evaluation -= board_value[i][j]
        return evaluation

    def str_value(self, value):
        if value == self.MAXIMUM:
            return "Infinity"
        elif value == self.MINIMUM:
            return "-Infinity"
        else: return value

    def write_traverse_log(self, board, alpha, beta):
        move = board.move
        value = board.value
        depth = board.depth         
        alpha = self.str_value(alpha)
        beta = self.str_value(beta)
        value = self.str_value(value)
        if move == None:
            # print 'is root'
            self.traverse_log.append('root,{},{},{},{}'.format(depth, value, alpha, beta))
            print 'root,{},{},{},{}'.format(depth, value, alpha, beta)
        elif isinstance(move, str):
            self.traverse_log.append('{},{},{},{},{}'.format(move, depth, value, alpha, beta))
            print '{},{},{},{},{}'.format(move, depth, value, alpha, beta)
        elif depth == 0:
            self.traverse_log.append('root,{},{},{},{}'.format(depth, value, alpha, beta))
            print 'root,{},{},{},{}'.format(depth, value, alpha, beta)
        else :
            self.traverse_log.append('{}{},{},{},{},{}'.format(mapping[move[0][1]], move[0][0] + 1, depth, value, alpha, beta))
            print '{}{},{},{},{},{}'.format(mapping[move[0][1]], move[0][0] + 1, depth, value, alpha, beta)


    def write_log(self):
        f = open("output.txt", "a")
        for i, line in enumerate(self.traverse_log):
            print line
            if i == len(self.traverse_log) - 1:
                f.write(line)
            else: 
                f.write(line + "\n")
        f.close()

    def write_all(self, board):
        f = open('output.txt', 'w')
        s = board.board_str() 
        f.write(s + '\n')
        for i, line in enumerate(self.traverse_log):
            print line
            if i == len(self.traverse_log) - 1:
                f.write(line)
            else: 
                f.write(line + "\n")
        f.close()


    def run(self):
        root = Board('root', 0, self.board, self.player)
        root.parent = root
        root.next = root
        self.max_value(root, self.MINIMUM, self.MAXIMUM)
        n = root.next
        print n
        # n.write_board()
        # self.write_log()
        self.write_all(n)


    def get_children(self, board):
        moves = board.get_valid_moves()
        children = list()
        for move in moves:
            new_board = board.make_move(move)
            children.append(new_board)
        return children

    def max_value(self, board, alpha, beta):
        # print 'board depth is ', board.depth
        children = self.get_children(board)
        print 'Turn --  -- - ', board.player 
        opponent_board = Board('pass', board.depth + 1, board.board, board.opponent) 
        # opponent_board.value = self.evaluate(opponent_board)

        opponent_children = self.get_children(opponent_board)

        # if board.depth == self.max_depth or board.is_game_over() or \
                # board.move == 'pass' and board.parent.move == 'pass':
        if board.depth == self.max_depth or board.move == 'pass' and board.parent.move == 'pass':
            # print board
            board.value = self.evaluate(board)
            self.write_traverse_log(board, alpha, beta)
            # print 'board depth ', board.depth
            # print 'weird value', board.value
            # print 'o1'
            return board.value 

        board.value = self.MINIMUM
        self.write_traverse_log(board, alpha, beta)
        # print 'o2'

        if len(children) == 0 and len(opponent_children) >= 0:
            opponent_board.parent = board
            children.append(opponent_board)
            
        for child in children:
            print 'here'
            board.value = max(board.value, self.min_value(child, alpha, beta))
            child.parent = board
            
            if board.value >= beta:
                self.write_traverse_log(board, alpha, beta)
                # print 'board depth', board.depth
                # print 'o3'
                # print 'player is ', board.player
                # print 'weird value', board.value
                # print board
                return board.value
            else:
                if board.value > alpha:
                    board.next = child
                alpha = max(alpha, board.value) 
            # print 'bd', board.depth
            self.write_traverse_log(board, alpha, beta)
            # print 'o4'
        return board.value


    def min_value(self, board, alpha, beta):
        children = self.get_children(board)
        print 'Turn --  -- - ', board.player 
        # print 'bbddd', board.depth

        opponent_board = Board('pass', board.depth + 1, board.board, board.opponent) 
        # opponent_board.value = self.evaluate(opponent_board)
        opponent_children = self.get_children(opponent_board)
        # if board.depth == self.max_depth or board.is_game_over() or \
                # (board.move == 'pass' and board.parent.move == 'pass'):
        if board.depth == self.max_depth or board.move == 'pass' and board.parent.move == 'pass':
            print 'maximum depth or two pass'
            # print board
            board.value = self.evaluate(board)
            # print 'o5'
            # print 'board depth',board.depth
            # print 'player is ', board.player
            self.write_traverse_log(board, alpha, beta)
            # print 'weird value', board.value
            return board.value 

        board.value = self.MAXIMUM
        self.write_traverse_log(board, alpha, beta)
        # print 'o6'

        if len(children) == 0 and len(opponent_children) >= 0:
            opponent_board.parent = board
            children.append(opponent_board)

        for child in children:
            board.value = min(board.value, self.max_value(child, alpha, beta))
            child.parent = board

            if board.value <= alpha:
                self.write_traverse_log(board, alpha, beta)
                # print 'o7'
                # print 'weird value', board.value
                # print 'board depth ', board.depth
                # print board
                return board.value
            else:
                if board.value < beta:
                    board.next = child
                beta = min(beta, board.value) 
            self.write_traverse_log(board, alpha, beta)
            # print 'o8'

        return board.value

   
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "i:")
    # print opts, args
    # parse = Parse()
    paramters = {}
    for opt, arg in opts:
        if opt == '-i':
            # print arg
            input_file = open(arg)
            player = input_file.readline().strip()
            # print 'max player: {}'.format(player)
            depth = int(input_file.readline().strip())
            # print 'cut off depth: {}'.format(depth)
            board = list()
            for i in range(8):
                board.append([])
                for value in input_file.readline().strip():
                    board[i].append(value)
            agent = AlphaBetaPruning(board, depth, player)
            agent.run()

        else:
            sys.exit(2)


