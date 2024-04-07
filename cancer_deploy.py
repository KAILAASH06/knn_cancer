import streamlit as st
import pickle
from sklearn.metrics.pairwise import euclidean_distances



def get_gene_expression1():
    gene1 = st.text_input("gene1")
    return gene1

def get_gene_expression2():
    gene2 = st.text_input("gene2")
    return gene2


def predict_cancer(g1,g2):
    loaded_model = pickle.load(open('knn_model.pkl','rb'))
    new_data = [[float(g1),float(g2),]]    
    prediction = loaded_model.predict(new_data)
    st.write("Prediction with new data: ")
    st.write(prediction)
    if prediction == 1:
        st.write("Cancer detected")
    else:
        st.write("No cancer")
    




if __name__ == "__main__":
    st.title('Cancer detection using knn')   
    gene_expression1 = get_gene_expression1()
    gene_expression2 = get_gene_expression2()
    st.write("The parameters you entered are: ")
    st.write("geneexpression1 ", gene_expression1)
    st.write("geneexpression2 ", gene_expression2)
    
    

if st.button("Predict"):
    predict_cancer(gene_expression1,gene_expression2)
    
