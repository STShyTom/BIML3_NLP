import torchtext
#torchtext.disable_torchtext_deprecation_warning()

def load_file(file):
    with open(file, 'r') as f:
        return f.read()

if __name__ == '__main__':
    fichier = load_file('dataset/train.txt')
    liste_lignes = fichier.split('\n')
    liste_textes, liste_emotions = [], []
    for ligne in liste_lignes:
        if ligne:
            texte, emotion = ligne.split(';')
            liste_textes.append(texte)
            liste_emotions.append(emotion)
