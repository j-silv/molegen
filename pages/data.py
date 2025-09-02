import streamlit as st
from molegen.data import DataframeDataset
from rdkit import Chem
import random
from collections import defaultdict

@st.cache_resource(show_spinner="Fetching ZINC dataset...")
def get_data(**kwargs): 
    """Create cached instance of MoleGen dataset (contains dataframe)"""
    return DataframeDataset(**kwargs)


def init_session_state(df):
    """Initialize the session state for a dynamic app experience
    
    idx : currently selected row idx in the dataframe (ignored if user overrides smiles string)
    smiles     : currently selected smiles string (determined from idx)
    mol        : currently selected mol (determined from idx)
    """
    
    if 'idx' not in st.session_state:
        st.session_state['idx'] = 0
        
    if 'smiles' not in st.session_state:
        st.session_state['smiles'] = df['canonical_smiles'].iloc[st.session_state['idx']]  

    if 'mol' not in st.session_state:
        st.session_state['mol'] = df['mol'].iloc[st.session_state['idx']]          
        
def callback_random(data):
    """Get next mol img but keep sampling if we get the same as previous"""
    
    idx = st.session_state['idx']
    while idx == st.session_state['idx']:
        idx = random.randrange(data.n_samples)
    
    # we need to update this state in case user has overriden it earlier
    st.session_state['idx'] = idx
    st.session_state['smiles'] = data.df['canonical_smiles'].iloc[idx] 
    st.session_state['mol'] = data.df['mol'].iloc[idx] 


@st.dialog("Invalid input")
def dialog_invalid_smiles():
    """Pop up to tell user that SMILES string is invalid
    
    TODO: replace this with just displaying an invalid molecule image
    in the image part of the app
    """
    
    st.write("Invalid SMILES string, please try again")


def callback_text_input():
    """Validate SMILES string"""
    
    mol = Chem.MolFromSmiles(st.session_state['smiles'])
    
    if mol == None:
        print("Invalid SMILES string")
        dialog_invalid_smiles()
    else:
        st.session_state['mol'] = mol

                

def view_options():
    """Add options for modifying data visualization"""
    
    # for some reason i have to specify 'bottom' otherwise right option is higher than left
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.checkbox("Dark mode", value=False, key='darkmode')
        st.checkbox("Monochrome", value=False, key='bw')

    
def molecule_select(data):
    """Selects the molecule with SMILES string and random idx selector"""
    
    st.text_area("**SMILES string**", height=68, key="smiles", on_change=callback_text_input)
    get_img(data)
    st.button("Random ZINC sample", on_click=callback_random, args=[data])   

        
def get_img(data):
    """Update image on canvas"""
    
    data.init_draw(darkmode=st.session_state['darkmode'], bw=st.session_state['bw'])
    df_img = data.get_img(st.session_state['mol'])
    st.image(df_img)
    

def feature_display(data):
    """Display all node and edge features"""
    
    st.write("**Atom to token mapping**")
    st.code(data.n2t, wrap_lines=True)
    
    st.write("**Bond to token mapping**")
    st.code(data.e2t, wrap_lines=True)
    
    
    mol = st.session_state['mol']

    st.write("Bag of atoms")
    token_boa = data.get_boa(mol)[0, :].tolist()
    boa = {data.t2n[token]: token_boa[token] for token in data.t2n if token_boa[token] > 0}
    st.code(boa, wrap_lines=True)
        
    # st.write("Bond connectivity") 
    # TODO: convert into human readable tuples where we have "(atom1, atom2)" to indicate an edge       
    edge_index, edge_attr = data.get_connectivity(mol)
    # st.code(edge_index, wrap_lines=True)
    
    st.write("Bond features")
    edge_attr = data.convert_edge_features(edge_attr).tolist()
    decoded_edge_attr = defaultdict(int)
    for token in edge_attr:
        decoded_edge_attr[data.t2e[token]] += 1
    # divide by 2 because we have undirected edges and double count
    for k in decoded_edge_attr:
        decoded_edge_attr[k] = decoded_edge_attr[k] // 2
        
    decoded_edge_attr = dict(decoded_edge_attr)
    st.code(decoded_edge_attr, wrap_lines=True)
    
    
    
def main():
    """Define layout of app"""
    
    st.subheader("SMILES viewer")

    data = get_data()
    init_session_state(data.df)

    col_molselect, col_descript = st.columns(2, border=True)
    
    with col_molselect:
        view_options()
        molecule_select(data)

    with col_descript:
        feature_display(data)
        
    

if __name__ == "__main__":
    main()