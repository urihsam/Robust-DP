from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector


multici = importr("MultinomialCI")

def isRobust(prob, std, attack_size):
    fv = FloatVector(sorted(prob)[::-1])
    ci = np.array(multici.multinomialCI(fv, 0.95))
    qi = ci[0,0]
    qj = ci[1,1]
    alpha = np.linspace(1.01,2.0, 100)
    # pdb.set_trace()
    bound = (-np.log(1-qi-qj+2*((qi**(1-alpha)+qj**(1-alpha))/2)**(1/(1-alpha)))/alpha).max() #Lemma 1
    # return np.sqrt(bound*2.*std**2)
    if bound > attack_size**(2.)/2./std**(2.): # Theorem 2
        return np.array([True, np.sqrt(bound*2.)*std])
    else:
        return np.array([False, np.sqrt(bound*2.)*std])