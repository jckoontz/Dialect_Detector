from __future__ import division
import nltk, re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from math import log
#read text
arg = open('ElSecreto.txt','rb').read().decode('utf-8')
mex = open('YTuMamaTambien.txt','rb').read().decode('utf-8')
esp = open('Biutiful.txt','rb').read().decode('utf-8')

#Tokenize texts
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
arg_tokens = tokenizer.tokenize(arg)
mex_tokens = tokenizer.tokenize(mex)
esp_tokens = tokenizer.tokenize(esp)
punctuation = re.compile(r'[-.?¿!¡,":;#()|0-9<>]')

#Preprocess Argentina Text
#punctuation = re.compile(r'\w+')
for i in range(len(arg_tokens)): 
	#tokens[i] = re.sub(r'\d',"", tokens[i])
	arg_tokens[i] = re.sub(r'\d',"", arg_tokens[i])
	arg_tokens[i] = punctuation.sub("",arg_tokens[i])
	arg_tokens[i] = arg_tokens[i].lower()

tokens_arg = [token for token in arg_tokens if token != ""]
#for token in tokens_arg: 
	#print(token)

#Preprocess Argentina Text                                                                                  
for i in range(len(mex_tokens)):                                                  
	mex_tokens[i] = re.sub(r'\d',"", mex_tokens[i])
	mex_tokens[i] = punctuation.sub("", mex_tokens[i])                                               
	mex_tokens[i] = mex_tokens[i].lower()

tokens_mex = [token for token in mex_tokens if token != ""]
#for token in tokens_mex: 
	#print(token)
#Preprocess Spanish Text:
for i in range(len(esp_tokens)):
        esp_tokens[i] = re.sub(r'\d',"", esp_tokens[i])
        esp_tokens[i] = punctuation.sub("", esp_tokens[i])
        esp_tokens[i] = esp_tokens[i].lower()
tokens_esp = [token for token in esp_tokens if token != ""]
for token in tokens_esp:
	print(token)

#for token in tokens: 
	#print token.encode('utf-8')
argentina_tokens = set(tokens_arg)
#for token in movie1_tokens:                                                                                
	#print token
 #print movie1_tokens.encode('utf-8')    

#Creat Set of Mex tokens
mexico_tokens = set(tokens_mex)

#Create Set of Esp tokens
spain_tokens = set(tokens_esp)
#Generate the intersection of Mexican and Argentine Spanish
print("We can generate the instersection from the three sets, Argentine, Mexican, and Castilian Spanish, in the following way:")
#print(set.intersection(movie1_tokens, movie2_tokens))
print("To calculate the Jaccard coefficient we divide the length of the intersection of the sets of types by the length of the union of these sets:")
lenIntersect = len(set.intersection(argentina_tokens, spain_tokens))
lenUnion = len(set.union(argentina_tokens, spain_tokens))
print(lenIntersect/lenUnion)



#Caluclate the probability of picking a text from Spain, Argentina, or Mexico                
total = len(arg_tokens) + len(mex_tokens) + len(esp_tokens)
print(total)

ArgP = len(arg_tokens) / total
MexP = len(mex_tokens) / total
EspP = len(esp_tokens) / total

print("probability to pick Argentina:", ArgP)
print("probability to pick Mexico:",MexP)
print("probability to pick Spain:",EspP)


#for token in movie2_tokens:
	#print token

#Frequency profile from the token-list
fpA = Counter(arg_tokens)
#print("Frequency profile of Argentine Spanish:")
#print(fpA)

#Frequency profile from the Mexico token-list                                  
fpM = Counter(mex_tokens)
#print("Frequency profile of Mexican Spanish:")
#print(fpM)

#FP of Castilian Spanish
fpE = Counter(esp_tokens)
#print("Frequency profile of Argentine Spanish:")
#print(fpE)


#We will need the total token count to calculate the relative frequency of the tokens, that is to generate likelihood estimates. We could brute force add one to create space in the probability mass for unknown tokens.
totalArg = sum(fpA.values()) + 1
totalMex = sum(fpM.values()) + 1
totalEsp = sum(fpE.values()) + 1
print("total Argentina counts + 1:", totalArg)
print("total Mexico counts + 1:", totalMex)
print("total Spain counts + 1:", totalEsp)



#Relavatize the counts in the frequency profiles
fpA = Counter( dict([ (token, frequency/totalArg)  for token, frequency in fpA.items() ]) )
fpM = Counter( dict([ (token, frequency/totalMex)  for token, frequency in fpM.items() ]) )
fpE = Counter( dict([ (token, frequency/totalEsp)  for token, frequency in fpE.items() ]) )

#We can now compute the default probability that we want to assign to unknown words as $1 / totalSpam$ or $1 / totalHam$ respectively. Whenever we encounter an unknown token that is not in our frequency profile, we will assign the default probability to it.
defaultArg = 1 / totalArg
defaultMex = 1 / totalMex
defaultEsp = 1 / totalEsp
print("default Argentina probability:", defaultArg)
print("default Mexico probability:", defaultMex)
print("default Spain probability:", defaultEsp)

#Unknoen text
unknown = open('Volver.txt','rb').read().decode('utf-8')
unknown_tokens = tokenizer.tokenize(unknown)
for i in range(len(unknown_tokens)):
        #tokens[i] = re.sub(r'\d',"", tokens[i])                                            
        unknown_tokens[i] = re.sub(r'\d',"", unknown_tokens[i])
        unknown_tokens[i] = punctuation.sub("",unknown_tokens[i])
        unknown_tokens[i] = unknown_tokens[i].lower()
tokens_unknown = [token for token in unknown_tokens if token != ""]

#Since this number is very small, a better strategy might be to sum up the log-likelihoods:
result_Arg = 0.0
for token in tokens_unknown:
    result_Arg += log(fpA.get(token, defaultArg), 2)
result_Arg += log(ArgP)
print(result_Arg)

#Mexico
result_Mex = 0.0
for token in tokens_unknown:
    result_Mex += log(fpM.get(token, defaultMex), 2)
result_Mex += log(MexP)
print(result_Mex)

#Spain
result_Esp = 0.0
for token in tokens_unknown:
    result_Esp += log(fpE.get(token, defaultEsp), 2)
result_Esp += log(EspP)
print(result_Esp)
#print(tokens_unknown) 
#print(fpA)

#

if max(result_Mex, result_Esp, result_Arg) == result_Esp:
    print("text is from Spain")
elif max(result_Mex, result_Arg) == result_Mex:
    print("text is from Mexico")
else: 
    print("text is from Argentina")
#else:
	#print("text is from Argentina")

#elif max(result_Mex, result_Esp) == result_Esp:
 #    print("text is from Spain")
#else: 
 #   print("text is from Mexico")

#Argentina: Create model using the token, the token frequency, and the length of the token
#model = [ (i, fp[i], len(i)) for i in fp ]
#print(model)

#Tab-delimited format of the model
#for x in model:
	#print "\t".join( (str(x[1]), str(x[2]), x[0]) ) 
 


#Create Meixco model using the token, the token frequency, and the length of the token                                             
#modelM = [ (i, fp[i], len(i)) for i in fpM ]
#print(modelM)

#Meico Tab-delimited format of the model                                                                                  
                                                                                                            


