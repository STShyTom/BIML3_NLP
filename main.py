

def load_file(file):
    with open(file, 'r') as f:
        return f.read()

if __name__ == '__main__':
    liste_textes = load_file('dataset/train.txt')