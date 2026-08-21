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

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

K_transposed = K.transpose(-2, -1)
print("K transposed shape:", K_transposed.shape)

scores = Q @ K_transposed

print("scores shape:", scores.shape)
print(scores)


"""
Output:
Q shape: torch.Size([1, 2, 4, 8])
K shape: torch.Size([1, 2, 4, 8])
V shape: torch.Size([1, 2, 4, 8])
K transposed shape: torch.Size([1, 2, 8, 4])
scores shape: torch.Size([1, 2, 4, 4])
tensor([[[[ 2.1655, -1.4713,  2.5439,  3.1282],
          [ 2.1704,  5.6811, -3.5576,  2.4307],
          [-1.8506, -0.3029, -1.2479,  7.0405],
          [-1.3146, -3.2209, -0.5854,  7.3494]],

         [[ 0.6638,  1.5666,  0.2625, -0.7211],
          [ 0.6177, -1.5074, -1.1462,  0.1730],
          [-2.4481,  1.8047,  3.0576, -0.4164],
          [ 2.8584, -1.5597, -2.8155, -1.6604]]]])

two for the two heads, each table is query tokens on the left
and key tokens on the right with the scores for each 
higher score means query token 1 matched key token 4 the most

but that could be a problem because if tokens are generated left to right
token 1 shouldnt be allowed to look at token 4, because token 4 is in the future

next step is casual masking
should be like this:
token 1 can look at token 1
token 2 can look at token 1,2
token 3 can look at token 1,2,3
token 4 can look at token 1,2,3,4

create a mask to block the upper right part of the 4x4 table
"""

mask = torch.tril(torch.ones(T, T))

print("mask shape:", mask.shape)
print(mask)

scores_masked = scores.masked_fill(mask == 0, float("-inf"))

print("masked scores shape:", scores_masked.shape)
print(scores_masked)

'''
torch.ones(T, T)
creates a 4by4 table of ones
torch.tril(...) keeps only the lower triangle
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
then 
scores.masked_fill(mask == 0, float("-inf"))
replaces blocked positions with negative infinity
does that because later softmax turns scores into probabilities, very large negative
becomes prob 0, so blocked get ignored

Output:
mask shape: torch.Size([4, 4])
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
masked scores shape: torch.Size([1, 2, 4, 4])
tensor([[[[-0.5739,    -inf,    -inf,    -inf],
          [ 0.1315,  3.3302,    -inf,    -inf],
          [-0.7941, -2.0308, -5.3198,    -inf],
          [-1.3189, -1.5536, -3.6209,  4.6663]],

         [[-0.8863,    -inf,    -inf,    -inf],
          [-4.6488, -1.1183,    -inf,    -inf],
          [-2.2890, -2.4832,  7.6423,    -inf],
          [-3.6928, -2.9871, -2.1283, -0.5395]]]])
'''
