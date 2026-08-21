''''Three number representations for each token:
Q = query
k = key
v = value



#B = batch size
- how many separate text examples are being processed at once
#H = number of attention heads
- transformers do attention multiple times in parallel using different "heads"
- each head learns to pay attention to different patterns
- one head might care about nearby words, another might care about grammar
- another might care about names or references
#T = sequence length or number of tokens
- how many tokens are in the sequence
#D = head dimension
- each token is represented by a vector of numbers
- because multiple heads, each head gets a slice of that representation
- slice size = D
- D = 8 means that each token, inside each head, is represented by 8 numbers


Attention asks: for this word, which earlier words matter?
"The trophy would not fit in the suticase because it was too big."
- it could be trophy or suitcase
- attention connects it -> trophy


what is head?
- single attention head = one way of looking at the sentence
- language has many different relationships happening at once
- model migth need to track:
-   who did the action
-   what object is being referred to
-   which adjective describes which noun
-   which words are nearby
-   which earlier phrase sets the topic

- one attention head cant capture all of these:
-   transformers use multiple heads
- each head looks for different things
- they run in parallel

attention needs to know:
- which example are we processing? B
- which attention head is doing the looking? H
- which token is being processed? T
- what is the numeric representation of that token? D

"for every example, for every head, compare every token to other tokens using their numeric vectors

example:

I like pizza

tokens = ["I", "like", "pizza"] = T = 3
two attention heads:
- head 1 might look at grammar
- head 2 might look at meaning H = 2

for each token in each head, model stores small vector
- D = how many numbers per vector D = 4

only once sentence = B = 1
- greater than B = 1 = more examples + different sentences


'''
#import the pytorch library
import torch


B = 1 #one sentence
H = 2 #different heads / views/ look at meaning, grammar, syntax, etc
T = 4 #4 token positions (4 tokens)
D = 8 #how many numbers per vector


Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
V = torch.randn(B, H, T, D)


#Q, K, V become 3 fake tensors full of random numbers
#each have the shape of:
#2 sentences × 4 heads × 8 tokens × 16-dimensional vector

#Q = query vectors, K = key vectors, V = value vectors

#dummy numbers to understand attention math

#next operation is Q compared with K
#produces attention scores !

#after doing that, shape changes to this
# Q [B, H, T, D]
# K [B, H, T, D]
#scores: [B, H, T, T]

#why [T, T] !!!
 
#Because each token compares itself to every token
#for ex. if T = 4, each head creates a 4 by 4 table


'''
            key token 1   key token 2   key token 3   key token 4
query 1      score         score         score         score
query 2      score         score         score         score
query 3      score         score         score         score
query 4      score         score         score         score
'''

# ^^^ this is for t = 4

#for each token, how much does it care about each other token


