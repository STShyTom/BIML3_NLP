from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from RNN import RNN

'''Chargement des données d'un fichier texte'''
def chargement_fichier(fichier):
    with open(fichier, 'r') as f:
        return f.read()

'''Préparation des données par encodage one-hot'''    
def yield_tokens(texte):
    # Création d'une correspondance entre les mots et les indices
    for ligne in texte:
        yield ligne.split()

def listOfListOfLine(texte):
    return [line.split() for line in texte]

'''Recherche de la taille maximale des textes'''     
def findMaxSize(liste_textes):
    list_listTextes = []
    for lineTexte in liste_textes:
        list_listTextes.append(lineTexte.split())
        
    return max([len(ligne) for ligne in list_listTextes])
    
'''Bourrage des textes pour qu'ils aient tous la même taille'''
def bourrage(liste_textes):
    taille_max = findMaxSize(liste_textes)
    liste_resultat = []
    for line in yield_tokens(liste_textes):
        if len(line) < taille_max:
            line.extend(['#'] * (taille_max - len(line)))
        liste_resultat.append(' '.join(line))
    return liste_resultat

def stringToEncode(listString, vocab):
    return [vocab[token] for token in listString]


if __name__ == '__main__':
    fichier = chargement_fichier('dataset/train.txt')
    liste_lignes = fichier.split('\n')
    # Création des listes de textes et d'émotions
    liste_textes, liste_emotions = [], []
    for ligne in liste_lignes:
        if ligne:
            texte, emotion = ligne.split(';')
            liste_textes.append(texte)
            liste_emotions.append(emotion)
            
    # Bourrage des textes
    liste_textes = bourrage(liste_textes)

    # Création des vocabulaires pour les textes et les émotions
    vocab_texte_encodes = build_vocab_from_iterator(yield_tokens(liste_textes))
    vocab_emotions_encodes = build_vocab_from_iterator(yield_tokens(liste_emotions))
    # for token in texte_encodes.get_itos():
    #    print(f"{token} -> {texte_encodes[token]}")
    

    # Encodage des textes
    listofList = listOfListOfLine(liste_textes)
    encodeTextList = [stringToEncode(liste, vocab_texte_encodes) for liste in listofList]
    tensorEncodeText = torch.tensor(encodeTextList) # Encodage des textes sous forme de token en tensor
    
    # Encodage des émotions
    listofList = listOfListOfLine(liste_emotions)
    encodeEmotionList = [stringToEncode(liste, vocab_emotions_encodes) for liste in listofList]
    tensorEncodeEmotion = torch.tensor(encodeEmotionList) # Encodage des émotions sous forme de token en tensor
    # for token in vocab_emotions_encodes.get_itos():
    #     print(f"{token} -> {vocab_emotions_encodes[token]}")
      
    # Hyperparamètres
    batch_size = 1
    emb_size = 128
    hidden_size = 128
    combined_size = 256
    output_size = 6
    learning_rate = 0.001
    epochs = 100
    
    dataSetTrain = torch.utils.data.TensorDataset(tensorEncodeText, tensorEncodeEmotion)    
    loaderTrain = DataLoader(dataSetTrain, batch_size=batch_size, shuffle=True)
    
    # Création du modèle    
    modele = RNN(len(vocab_texte_encodes), emb_size, hidden_size, combined_size, output_size)
    
    # Fonction de coût et optimiseur
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(modele.parameters(), lr=learning_rate)
    
    # Entrainement
    hidden = modele.initHidden(batch_size)
    
    # for x,t in loaderTrain:
        
    #     #on encode en one hot
    #     xEncode = F.one_hot(x, len(vocab_texte_encodes)).int()
    #     tEncode = F.one_hot(t, len(vocab_emotions_encodes)).int()       
        
    #     #for i in range(x.size()):
    #         #output, hidden = modele(x[i], hidden)
    #         output, hidden = modele(x[0], hidden)
    #         loss = criterion(output, t)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    
    x,t = loaderTrain.__iter__().__next__()
    xEncode = F.one_hot(x[0][0], len(vocab_texte_encodes)).float()
    tEncode = F.one_hot(t, len(vocab_emotions_encodes)).float()
    # output, hidden = modele(xEncode, hidden)
    print("------------------------------------------------")
    print(xEncode.shape)
    print("------------------------------------------------")
    print(tEncode)
    # print("------------------------------------------------")
    # print(output)
    