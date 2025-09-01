import streamlit as st
from molegen.data import DataframeDataset

@st.cache_resource(show_spinner="Fetching ZINC dataset...")
def get_data(**kwargs): 
    return DataframeDataset(**kwargs)

def callback_prev_button():
    idx = st.session_state['df_mol_idx'] - 1
    
    if idx < 0:
        idx = st.session_state['df_num_mols'] - 1
    
    st.session_state['df_mol_idx'] = idx
    
def callback_next_button():
    idx = st.session_state['df_mol_idx'] + 1
    
    if idx > st.session_state['df_num_mols'] - 1:
        idx = 0
    
    st.session_state['df_mol_idx'] = idx

    
def main():
    st.subheader("Exploring the ZINC dataset")

    data = get_data()
    df = data.df

    col_molselect, col_descript = st.columns(2, border=True)

    if 'df_mol_idx' not in st.session_state:
        st.session_state['df_mol_idx'] = 0
        st.session_state['df_num_mols'] = len(data.df)
        
    mol = df['mol'].iloc[st.session_state['df_mol_idx']]

    with col_descript:
        st.write("**Atom to token mapping**")
        st.code(data.n2t, wrap_lines=True)
        st.write("**Bond to token mapping**")
        st.code(data.e2t, wrap_lines=True)
        
        st.write("Atom features")
        
        atom_features = data.get_node_features(mol)[:, 0].tolist()
        st.code(atom_features, wrap_lines=True)
    
    with col_molselect:
        
        darkmode = st.checkbox("Dark mode", value=False)
        bw = st.checkbox("Monochrome", value=False)
        
        data.init_draw(darkmode=darkmode, bw=bw)
            
        df_smiles = df['canonical_smiles'].iloc[st.session_state['df_mol_idx']]
        df_img = data.get_img(mol, legend="SMILES: " + df_smiles)

        
        image = st.image(df_img)
    
        buttons = st.container(horizontal=True, horizontal_alignment="left")
        buttons.button("Previous", on_click=callback_prev_button)
        buttons.button("Next", on_click=callback_next_button)

    
    

if __name__ == "__main__":
    main()