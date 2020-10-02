#Bibliotecas básicas
import pandas as pd
import numpy as np

#Bibliotecas visualização
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Bibliotecas especificas
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import umap
#****************************************************************************************

#########################
#   INSTRUCOES GERAIS   #
#########################

#1) Salvar os arquivos no mesmo local deste script.py:
#	a) base_completa.db
#	b) kmean.csv
#2) Verificar se as bibliotecas utilizadas estão instalados
#3) Verificar se o streamlit está instalado
#4) No prompt de comando verificar se a pasta atual é a mesma do script
#5) No prompt de comando digitar "streamlit run streamlit_cluster.py"



###############
#   SUMARIO   #
###############

#1-Extracao dos dados
#2-Cabecalho
#3-Grafico de Radar
#4-Visualizacao dos cluster
#5-PCA e UMAP
#6-Grafico Treemap
#7-Conclusao
#8-Jaba


#****************************************************************************************
#DEFINICOES GERAIS
pd.options.display.float_format = "{:,.2f}".format

DADOS_BASICOS = 'base_completa.db'
DADOS_KMEANS = 'kmeans.csv'
#DADOS_GMM = 'GMM.csv'
ESCALA = MinMaxScaler()
ESCALA_RADAR_MAX = 10
SEED = 42
EIXOS = 3

#****************************************************************************************

####################################
#   PARTE 1 - EXTRACAO DOS DADOS   #
####################################

#Base 1: cadastro basico dos parlamentares
df_basico=pd.read_csv('dados.csv', sep=';',
 		              dtype={'CPF':str,
	                         'SQ_CANDIDATO':str,
	                         'id':str,
	                         'ID_CAMARA':str})

#Base 2: clusters estimados por K-Means
df_kmeans = pd.read_csv(DADOS_KMEANS,sep=';',usecols=['cluster'],dtype={'cluster':int})
df_kmeans.rename(columns={'cluster':'kmeans'}, inplace=True)

#Base 3: clusters estimados por GMM
#df_gmm = pd.read_csv(DADOS_GMM ,sep=';',usecols=['cluster'],dtype={'cluster':int})
#df_kmeans.rename(columns={'cluster':'gmm'}, inplace=True)

#Juntando Base1 + Base2 + Base3 e ajustes
df_basico = pd.concat([df_basico,df_kmeans],axis =1)
#df_basico = pd.concat([df_basico,df_kmeans, df_gmm],axis =1)
df_basico['quantidade']=1
df_basico.fillna(0, inplace=True)
df_basico.drop_duplicates(inplace=True)


###########################
#   PARTE 2 - CABECALHO   #
###########################

st.title('Quem disse que político é tudo igual?')

st.markdown(
'Aliando Machine Learning e dados públicos, o projeto avaliou se a concepção de que *"político é tudo igual"* faz sentido sob a ótica da Ciência de Dados. Após a aplicação de algoritmos não-supervisionados de clustering, o projeto obteve êxito na identificação de perfis distintos de atuação entre os deputados federais em exercício.'
)
st.write('')
st.write('')

###############################
#   PARTE 3 - GRÁFICO RADAR   #
###############################

#PARTE 3.1 - DADOS
    
#separar a tabela
df = df_basico[['PERC_PRESENCA',
              'followers_count',
              'GASTO_GABINETE',
              'TOTAL_PROPOSTAS',
              'ORGAO_PARTICIPANTE',
              'ORGAO_GESTOR',
              'kmeans']]
df.drop_duplicates(inplace=True)              
df.sort_values(by='kmeans', inplace=True)  

#sera usado em seguida
df3=pd.DataFrame()

#aplicando um for poderoso que vai criar a tabela a ser aplicada no grafico de aranha
for i in df['kmeans'].unique():

    #separando em cluster
    df1 = df[df['kmeans']==i].drop(columns=['kmeans'])
    
    #aplicando a mediana em cada cluster
    ar1 = df1.median() 
    
    #transpondo o resultado para melhor leitura
    df2 = pd.DataFrame(ar1).T
    
    #juntando os grupos
    df3 = pd.concat([df3,df2],axis=0)
    
#aplicando a escala definida no inicio do script
ar2 = ESCALA.fit_transform(df3)*ESCALA_RADAR_MAX


#Alteracoes na metodologia de pontuacao dos clusteres devem ser aletrados no cabecalho.
#16/09: primeiramente as notas foram escalas entre 0 a 10 pelo metodo de minimos e
#maximos e depois segregados entre cluster.

#ultimos ajustes
df4 = pd.DataFrame(ar2).rename(columns={0:'Presença',
                                       1:'Twitter',
                                       2:'Gastos',
                                       3:'Propostas',
                                       4:'Órgãos participantes',
                                       5:'Órgãos geridos'})
df4.reset_index(drop=True, inplace=True)

#salvando o arquivo para ser usado no outro streamlit
df4.to_csv('var_escala0a10_kmeans.csv', sep=';')


#PARTE 3.2 - GRAFICO ARANHAS CLUSTER 0

st.header('Apresentando os clusters identificados')

#st.write(
#'Avaliando o desempenho parlamentar em 2019, identificamos 5 diferentes perfis de atuação. Os gráficos abaixos revelam uma comparação entre os perfis.'
#)



explicacao ={
	'Presença: percentual de presença em sessões legislativas em 2019',
	'Twitter: número de seguidores no Twitter registrados em agosto/2020',
	'Gastos: gastos de gabinete registrados em 2019',
	'Propostas: total de propostas enviadas pelo parlamentar em 2019',
	'Órgãos participantes: total de órgãos legislativos o qual o parlamentar apenas participa',
	'Órgãos geridos: total de órgãos legislativos geridos pelo parlamentar'
}

#funcao grafico principal de aranha
def grafico_aranha(cluster,cor):
    
    #convertendo para np.array os parametros a serem utilizados no grafico
    valores = np.array(df4.iloc[cluster,:])
    nomes = np.array(df4.columns)

    #plota o grafico
    #st.header('Cluster {}'.format(cluster))
    fig = px.line_polar(df4,
                        r = valores,
                        theta = nomes,
                        line_close = True,
                        range_r = (0,10),
                        start_angle=90,
                        labels=explicacao,
                        color_discrete_sequence = cor
    )
    fig.update_traces(fill='toself')
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=False)

st.write('**Perfil "Muito Acima da Média"**: os deputados federais que superaram em muito a média nacional. Eles elaboram mais propostas, participam mais e gastam menos.')

#finalmente o grafico 
#ex.: cor = px.colors.qualitative.D3
grafico_aranha(4, px.colors.qualitative.Vivid)


#explicacao das variaveis
explic_var = '''O que significam essas variáveis?:
 - Propostas: total de propostas enviadas pelo parlamentar em 2019
 - Órgãos participantes: total de órgãos legislativos os quais o parlamentar participa
 - Órgãos geridos: total de órgãos legislativos geridos pelo parlamentar
 - Presença: percentual de presença em sessões legislativas em 2019
 - Twitter: número de seguidores no Twitter registrados em agosto/2020
 - Gastos: gastos de gabinete registrados em 2019'
'''
st.code(explic_var, language='python')


#PARTE 3.3 - GRAFICO ARANHAS CLUSTER RESTANTES

#Explicando os clusters
st.write('**Perfil "Acima da Média"**: os políticos que performaram acima da média porém com um maior gasto de gabinete.')
st.write('**Perfil "Medianos"**: os políticos que performaram nem tão bem, mas também nem tão mal.')
st.write('**Perfil "Abaixo da Média"**: aqueles que gastaram bastante e mesmo assim performaram mal. O Grupo conta com baixa participação em comissões, número de propostas, ao mesmo tempo que apresentam altos gastos de gabinete.')
st.write('**Perfil "Muito Abaixo da Média"**: os parlamentares com as piores performances. Apesar do grupo apresentar gasto de gabinete menor que os demais, eles são os que mais faltam e menos propõe.')
#abreviacao
nome_abr=['PRES','TWIT','GAST','PROP','ORG_P','ORG_G']

#o plano sera dividir a tela em 4 setores e fazer um grafico aranha em cada.
#infelizmente não usarei a funcao acima
fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}]*2]*2)

#Cluster 0
fig.add_trace(go.Scatterpolar(
      r=np.array(df4.iloc[0]),
      theta=np.array(nome_abr),
      fill='toself',
      name='Acima da Média'
),1,1)

#Cluster 3
fig.add_trace(go.Scatterpolar(
      r=np.array(df4.iloc[3]),
      theta=np.array(nome_abr),
      fill='toself',
      name='Medianos'
),1,2)

#Cluster 1
fig.add_trace(go.Scatterpolar(
      r=np.array(df4.iloc[1]),
      theta=np.array(nome_abr),
      fill='toself',
      name='Abaixos da Média'
),2,1)

#Cluster 2
fig.add_trace(go.Scatterpolar(
      r=np.array(df4.iloc[2]),
      theta=np.array(nome_abr),
      fill='toself',
      name='Muito Abaixo da Média'
),2,2)

#parametros de visualizacao
fig.update_traces(fill='toself')

fig.update_layout(
    autosize=False,
    width=550,
    height=550,
    showlegend=True,
  	polar=dict(
              angularaxis=dict(direction="clockwise"),
    	        radialaxis=dict(visible=False,range=[0, 10]),
              ),
    polar2=dict(
              angularaxis=dict(direction="clockwise"),
              radialaxis=dict(visible=False,range=[0, 10])
              ),
    polar3=dict(
              angularaxis=dict(direction="clockwise"),
              radialaxis=dict(visible=False,range=[0, 10])
              ),
    polar4=dict(
              angularaxis=dict(direction="clockwise"),
              radialaxis=dict(visible=False,range=[0, 10])
              )
    
)

st.plotly_chart(fig, use_container_width=True)


#PARTE 3.4 - TABELA FINAL

st.write('A mediana de cada cluster foi:')

#renomerando os indices para subir no streamlit
df3.index=['Acima da Média',
           'Abaixos da Média',
           'Muito Abaixo da Média',
           'Medianos',
           'Muito Acima da Média']

#ajuste de ordem (rever caso variaveis mudem)
df5 = df3.reindex(index=['Muito Acima da Média',
                         'Acima da Média',
                         'Medianos',
                         'Abaixos da Média',
                         'Muito Abaixo da Média'])

df5.rename(columns={'PERC_PRESENCA':'Presença',
                    'followers_count':'Twitter',
                    'GASTO_GABINETE':'Gasto',
                    'TOTAL_PROPOSTAS':'Propostas',
                    'ORGAO_PARTICIPANTE':'Partic. Órgãos',
                    'ORGAO_GESTOR':'Gerência Órgãos'},inplace=True)

st.table(df5.round(2))


##########################################
#   PARTE 4 - QUEM ESTA DENTRO DE CADA   #
##########################################
st.write('')
st.write('')
st.write('')
st.header('Visualizando os clusters')
st.write('Agora que entendemos a formação de cada cluster, vamos avaliar como os políticos se distribuem entre cada grupo.')

#configurando o unico dropdown do projeto para selecionar o cluster
lista_cluster = ('Muito Acima da Média',
                 'Acima da Média',
                 'Medianos',
                 'Abaixos da Média',
                 'Muito Abaixo da Média')

qual_cluster = st.selectbox(
     'Selecione o cluster',
     lista_cluster
     )

dict_cluster = {
    lista_cluster[0]:4, #muito acima da media e cluster 4
    lista_cluster[1]:0, #acima da media e cluster 0
    lista_cluster[2]:3, #mediano e cluster 3
    lista_cluster[3]:1, #abaixo da media e cluster 1        
    lista_cluster[4]:2, #muito abaixo da media e cluster 2
}

rev_dict_cluster = {value : key for (key, value) in dict_cluster.items()}

n_cluster = dict_cluster[qual_cluster]

#dicionario para as cores dos graficos. Tentei imitiar as cores usadas inicialmente
dict_cor = {
    0:'blue',
    1:'green',
    2:'purple',
    3:'red',
    4:'orange'
    }

#buscando os dados com a coordenadas de x e y (16x36)
df = pd.read_csv('graf.csv')
df.drop_duplicates(inplace=True) 

#informacoes gerais
st.write('{} de {} parlamentares'.format(len(df[df['kmeans']==n_cluster]),len(df)))

#o grafico
fig = go.Figure(data=go.Scatter(x=df['x'],
								y=df['y'],
								mode='markers',
								#marker_color=teste['kmeans'],
								text=df['NM_CANDIDATO'],
								marker=dict(
								    size=14,
                    color=np.where(df['kmeans'] == n_cluster, dict_cor[n_cluster], 'lightgray')
)))

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
				  width=800,
				  height=330,
				  margin=dict(l=0, r=0, t=0, b=0))
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig, use_container_width=False)


#########################################
#   PARTE 5 - VISUALIZACAO PCA E UMAP   #
#########################################

st.subheader('Representação em 3D dos clusters: PCA e UMAP')

#PARTE 5.1 - DADOS
#definindo os dados
df = df_basico[['ORGAO_PARTICIPANTE',
               'ORGAO_GESTOR',
               'TOTAL_PROPOSTAS',
               'PERC_PRESENCA',
               'GASTO_GABINETE',
               'followers_count'
              ]]
df.drop_duplicates(inplace=True) 


#aplicando ajuste de escala MinMax
df_sc = MinMaxScaler().fit_transform(df)

#reaplicando os nomes
df = pd.DataFrame(df_sc).rename({0:'ORGAO_PARTICIPANTE',
               	                 1:'ORGAO_GESTOR',
                	             2:'TOTAL_PROPOSTAS',
                   	             3:'PERC_PRESENCA',
                   	      	     4:'GASTO_GABINETE',
                   		         5:'followers_count'
                          	   },axis=1)

#PARTE 5.2 - PCA
#setando parametros iniciais
pca = PCA(n_components=EIXOS, random_state=SEED)

#usar o fit ou fit_transform?
embedding = pca.fit_transform(df)
#embedding = pca.fit(df)

#unificando o df reduzido + clusters
df_emb = pd.DataFrame(embedding)
result_pca = pd.concat([df_emb,df_basico['kmeans']], axis=1)


#PARTE 5.3 - UMAP

#definindo parâmetros do algoritmo
n_neighbors = 400       #(max=nrows-1) quanto maior mais preserva as características globais
min_dist= 1             #(max = 1) quanto menor, mais apertado fica cluster
n_components = EIXOS    #dimensões
metric = 'euclidean'    #tipo de distância (euclidiana é uma reta entre 2 pontos)
random_state= SEED

#definindo o redutor
reducer = umap.UMAP(n_neighbors = n_neighbors,
                    n_components=n_components,
                    min_dist= min_dist,
                    metric = metric,
                    random_state = random_state
                   )

#treinando o redutor de dimensionalidade do UMAP
embedding = reducer.fit_transform(df)
#embedding = reducer.fit(df)

#unificando o df reduzido + clusters
df_emb = pd.DataFrame(embedding)
result_umap = pd.concat([df_emb,df_basico['kmeans']], axis=1)


#PARTE 5.4 - GRAFICOS

#plostando os graficos de PCA e UMAP
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"},{"type": "scene"}]],
    subplot_titles=("PCA", "UMAP")
)

#plot do PCA
fig.add_trace(go.Scatter3d(x=result_pca[0],
                           y=result_pca[1],
                           z=result_pca[2],
                           mode="markers",
                           marker=dict(
                               size = 4,
                               color=np.where((result_pca['kmeans'] == n_cluster), dict_cor[n_cluster], 'lightgrey'),
                               opacity = 1
					        )                           
						),row=1, col=1
)

#plot do UMAP
fig.add_trace(go.Scatter3d(x=result_umap[0],
                           y=result_umap[1],
                           z=result_umap[2],
                           mode="markers",
                           marker=dict(
                               size = 4,
                               color=np.where((result_umap['kmeans'] == n_cluster), dict_cor[n_cluster], 'lightgrey'),
                               opacity = 1
					        )                           
						),row=1, col=2
)

fig.update_layout(margin=dict(l=50, r=0, b=0, t=0),
                  height=400,
                  width=700,
                  showlegend=False
                 )
st.plotly_chart(fig, use_container_width=False)


st.markdown(
'A visualização em 3 dimensões dos clusters é possível com a técnica de **redução de dimensionalidade**. Dos algoritmos disponíveis, o projetou adotou o **PCA** e o **UMAP**. Basicamente, ambos combinam as variáveis existentes, diminuindo o número de dimensões ao representar graficamente, porém enquanto o PCA o faz de maneira **linear**, o UMAP adota uma abordagem **não-linear**.'
)

#####################################
#   PARTE 6 - COMPARACAO DEPUTADO   #
#####################################

st.write('')
st.write('')
st.write('')
st.header('Comparando os políticos')
st.write('Finalmente, compararemos o desempenho dos parlamentares em relação ao seu cluster.')

#PARTE 6.1 - PARAMETROS INICIAIS
#selecionando o estado
lista=list(df_basico["SG_UE"].unique())
lista_uf=sorted(lista)
uf = st.selectbox("Selecione o estado",lista_uf, index=25)

#selecionando o parlamentar
lista=list(df_basico[df_basico["SG_UE"]==uf]["NM_PUBLICO"])
lista_parlamentar=sorted(lista)
nome = st.selectbox("selecione o parlamentar",lista_parlamentar, index=21)

#extraindo a imagem
id=df_basico[df_basico["NM_PUBLICO"]==nome]["ID_CAMARA"].iloc[0]
foto="https://www.camara.leg.br/internet/deputado/bandep/"+id+".jpg"
st.image(foto)

#criando a tabela basica a ser apresentada
df = df_basico[df_basico['NM_PUBLICO']==nome][['NM_CANDIDATO','kmeans','IDADE','SG_PARTIDO','VL_BENS','VR_DESPESA_CONTRATADA']]
df.rename(columns={
                   'NM_CANDIDATO':'Nome',
                   'kmeans':'Cluster', 
                   'IDADE':'Idade',
                   'SG_PARTIDO':'Partido',
                   'VL_BENS':'Patrim. declarado',
                   'VR_DESPESA_CONTRATADA':'Custo campanha eleitoral'}, inplace=True)
df.set_index('Nome', inplace=True, drop=True)
numero_cluster = df.iloc[0,0]
df.iloc[0,0] = rev_dict_cluster[numero_cluster]

st.subheader('Informações básicas')
st.table(df)


#PARTE 6.2 - CRIANDO DATAFRAME DE COMPARACAO
#dataframe do cluster

df_cluster = df3.iloc[numero_cluster]
df_cluster = pd.DataFrame(df_cluster).T

#dataframe do parlamentar
df_individual = df_basico[df_basico['NM_PUBLICO']==nome][['NM_PUBLICO',
                                                          'PERC_PRESENCA',
                                                          'followers_count',
                                                          'GASTO_GABINETE',
                                                          'TOTAL_PROPOSTAS',
                                                          'ORGAO_PARTICIPANTE',
                                                          'ORGAO_GESTOR']]
df_individual.set_index('NM_PUBLICO', inplace=True, drop=True)

#juntando ambos os dataframe + ajustes de nome
df_compara = pd.concat([df_cluster,df_individual],axis=0)
df_compara.rename(columns={
                            'PERC_PRESENCA':'Presença em sessões',
                            'followers_count':'Seguidores no Twitter',
                            'GASTO_GABINETE':'Gasto de gabinete',
                            'TOTAL_PROPOSTAS':'Propostas',
                            'ORGAO_PARTICIPANTE':'Partic. Órgãos',
                            'ORGAO_GESTOR':'Gerência Órgãos'},inplace=True)

#show me the money
st.subheader('Comparação entre cluster e parlamentar')
st.table(df_compara)


###########################
#   PARTE 7 - CONCLUSAO   #
###########################

st.write('')
st.write('')
st.write('')
st.header('Conclusão')
st.write('As evidências **sugerem a existência de diferentes perfis de atuação** dos deputados federais em 2019. Tanto os resultados dos algoritmos de clusterização quanto a investigação posterior dos dados permitiram a identicação de grupos distintos dentro da Câmara dos Deputados.'
) 
st.write('É pertinente a ressalva de que os perfis de atuação expostos acima foram definidos unicamente a partir das variáveis abordadas, de modo que outros conjuntos de variáveis poderão resultar em perfis de atuação distintos.')
st.write('Para saber mais sobre o projeto, seguem os links dos artigos do grupo. Enquanto o primeiro é sobre a definição do problema e a solução, o segundo aborda a técnica utilizada e a jornada.')
st.markdown('[Artigo sobre a Definição do Problema](https://medium.com/@rknagao/qual-%C3%A9-o-perfil-do-seu-candidato-a25890fe443a)',False)
st.markdown('[Artigo sobre a Jornada](https://medium.com/@guiazenha10/qual-%C3%A9-o-perfil-do-seu-candidato-p-02-990950c425b0?source=search_post---------0)',False)


######################
#   PARTE 8 - JABÁ   #
######################

st.header('Quem somos')
st.write(
  'A equipe Quarenteneiros foi formada durante o trabalho de conclusão do curso de Data Science & Machine Learning da Tera')

st.markdown('[Carlos Dias](https://www.linkedin.com/in/carlos-lima-dias/) - climadias@gmail.com',False)
st.markdown('[Guilherme Azenha](https://www.linkedin.com/in/guilherme-azenha/) - guiazenha10@gmail.com',False)
st.markdown('[Paulo Franco](https://www.linkedin.com/in/pffilho/) - paulo.franco.brsp@gmail.com',False)
st.markdown('[Rafael Kenji](https://www.linkedin.com/in/rafael-kenji-nagao/) - rknagao@gmail.com',False)


