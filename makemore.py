import torch
import matplotlib.pyplot as plt


# store the words
words = open('names.txt', 'r').read().splitlines()

# 28 by 28 array to store the 2 words and there frequency
N = torch.zeros((27,27), dtype= torch.int32)

# basically all the characters sorted [a,b,c,...]
chars = sorted(list(set(''.join(words))))

# stores the word and its number. eg {a:0, b:1,...}
stoi = {s:i+1 for i, s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

# we are also using special start and end characters
stoi['.']= 0


itos[0]= '.'

# store the frequency at the word position 
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      N[ix1, ix2] += 1

if __name__ == "__main__":

  # plt.figure(figsize=(16,16))
  # plt.imshow(N, cmap='Blues')
  # for i in range(27):
  #   for j in range(27):
  #     chstr = itos[i] + itos[j]
  #     plt.text(j,i, chstr, ha="center", va='bottom', color= 'gray')
  #     plt.text(j,i,N[i,j].item(), ha="center", va='top', color= 'gray' )
  # plt.axis('off')

  # plt.show()

  g = torch.Generator().manual_seed(2147483647)

  for i in range(50):
    out = []
    ix = 0
    while True:
      p = N[ix].float()
      p = p / p.sum()
      ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
      out.append(itos[ix])
      if ix == 0:
        break
    print(''.join(out))




  # print(sorted(b.items(), key = lambda kv : -kv[1]))