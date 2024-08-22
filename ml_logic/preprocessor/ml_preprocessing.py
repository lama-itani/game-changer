Chemin_Data_XLSX='../Data/Import_xls_ref.xlsx'
import numpy as np
import pandas as pd

def ML_clean_Data(fichier_Injure,fichier_Jeu,Chemin_Data_XLSX):

    # ----------------------------------------------------------------------------------------------------
    ### Etape 0 : Merge des tables playlist_data avec injury_data

    # injury_data avec uniquement 2 colonnes (DM_M1=1)
    fichier_blessure_binaire=fichier_Injure[['PlayerKey','DM_M1']]
    # Merge de la table playlist_data avec injury_data
    merged_df = pd.merge(fichier_Jeu, fichier_blessure_binaire,
                         on = 'PlayerKey',
                         how = 'left')
    # On renomme la colonne DM_M1, en Injured
    merged_df=merged_df.rename(columns={'DM_M1': 'Injured'})
    # On enleve les NA de la colonne Injured, pour avoir des 0 pour les non blessés
    merged_df.Injured=merged_df.Injured.fillna(0)

    # ----------------------------------------------------------------------------------------------------
    ### Etape 1 : Merge de la table merged_df (résultant de l'étape 0) et de la table fichier_Injure_drop_

    # On enlève 1 ligne de ce joueur qui a deux blessures au genou et cheville
    # pour ne pas avoir de doublon sur les données de la table left
    condition2 = fichier_Injure['GameID'] == '47307-10'
    condition3 = fichier_Injure['BodyPart'] == 'Ankle'
    fichier_Injure_drop = fichier_Injure[~(condition2 & condition3)]
    fichier_Injure_drop_=fichier_Injure_drop.drop(columns='PlayKey')
    # Merge de la table résultante de l'étape 0 avec la table fichier_Injure_drop_
    # Cela permet de rajouter le restant des colonnes de la table des blessures

    merged_df_2 = merged_df.merge(fichier_Injure_drop_,
                         on = ('PlayerKey','GameID'),
                         how = 'left')

    # ----------------------------------------------------------------------------------------------------
    ### Etape 2 : Feat Eng 1 : fillna de certaines colonnes et Construction de la colonne DM_TOT

    # Fillna sur les colonnes suivantes de la table mergée
    columns_1=['BodyPart','Surface','DM_M1','DM_M7','DM_M28','DM_M42','Weather','PlayType']
    for col in columns_1:
        merged_df_2[col]=merged_df_2[col].fillna(0)
    # Construction de la colonne DM_TOT, qui va de 1 à 4 selon la gravité de la convalescence
    merged_df_2['DM_TOT']=merged_df_2['DM_M1']+merged_df_2['DM_M7']+merged_df_2['DM_M28']+merged_df_2['DM_M42']

    # ----------------------------------------------------------------------------------------------------
    ### Etape 3 : Feat Eng 2 : Construction de la colonne DM_GR, selon la Gravité N/O : GN ou GO

    # Construction de la colonne DM_GR, selon la Gravité N/O : GN ou GO
    merged_df_2['DM_GR'] = merged_df_2['DM_TOT'].apply(lambda x: 'GN' if x <= 2 else 'GO')
    # On nettoie la Temperature en F, qui présente une valeur = -999
    merged_df_2['Temperature'] = merged_df_2['Temperature'].apply(lambda x: 0 if x <= -5 else x)
    # On transforme l'unité de la temperature de Far à Cel
    merged_df_2['Temperature_C'] = merged_df_2['Temperature'].apply(lambda x: round((x-32)*5/9,0))

    # ----------------------------------------------------------------------------------------------------
    ### Etape 4 : Feat Eng 3 : replace nan des colonnes StadiumType, PlayType, Position et PositionGroup

    merged_df_2['StadiumType']=merged_df_2['StadiumType'].replace(np.nan,'StadiumTypeNan')
    merged_df_2['PlayType']=merged_df_2['PlayType'].replace('Missing Data','PositionNan')
    merged_df_2['PlayType']=merged_df_2['PlayType'].replace('0','PositionNan')
    merged_df_2['Position']=merged_df_2['Position'].replace('Missing Data','PositionNan')
    merged_df_2['PositionGroup']=merged_df_2['PositionGroup'].replace('Missing Data','PositionGroupNan')

    # ----------------------------------------------------------------------------------------------------
    ### Etape 5 : Import des tables ref de formatage des données StadiumType et Weather

    StadiumType_sheet=pd.read_excel(Chemin_Data_XLSX,sheet_name='StadiumType')
    Weather_sheet=pd.read_excel(Chemin_Data_XLSX,sheet_name='Weather')

    # ----------------------------------------------------------------------------------------------------
    ### Etape 6 : merge des tables pour rendre plus lisibles les données Weather et StadiumType en les transformant

    # On enlève les espaces aux extremes de la colonne StadiumType, pour réussir la jointure de tables
    merged_df_2['StadiumType']=merged_df_2['StadiumType'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # On merge notre table pour la compléter de la table StadiumType_Sheet
    merged_df_3 = merged_df_2.merge(StadiumType_sheet,
                         on = 'StadiumType',
                         how = 'left')
    # On merge notre table pour la compléter de la table Weather_sheet
    merged_df_4 = merged_df_3.merge(Weather_sheet,
                         on = 'Weather',
                         how = 'left')

    # ----------------------------------------------------------------------------------------------------
    ### Etape 7 : Drop les colonnes determinées, pour diminuer la table finale en gardant les éléments gardés

    # On importe la table des colonnes des champs, et on indique OUI/NON sur la colonne DROP_INDICATEUR
    Drop_Colonne_sheet=pd.read_excel(Chemin_Data_XLSX,sheet_name='DropSheet')
    drop_table=Drop_Colonne_sheet[Drop_Colonne_sheet.DROP_INDICATEUR.str.contains("OUI")]
    # Liste des colonnes à droper
    colonnes_a_droper=list(drop_table['DROP_COLONNE'])
    # Table finale merge_df_5, qui possède les colonnes qu'on souhaite garder
    merge_df_5=merged_df_4.drop(colonnes_a_droper,axis=1)
    # On fait un dernier nettoyage sur les 0 de la colonne Weather_Visibilite
    merge_df_5['Weather_Visibilite']=merge_df_5.Weather_Visibilite.isna().fillna('neutral')

    # ----------------------------------------------------------------------------------------------------
    ### Etape 8 : On fait quelques sorties pour avoir des statistiques de vérification

    # Description des colonnes
    merge_df_5.describe(include='all').T
    print (merge_df_5.describe(include='all').T)
    print("✅ description")
    # Statistiques sur les valeurs par colonnes et détection des NAN si présent
    colonnes_du_data=list(merge_df_5.columns)
    for col in colonnes_du_data:
        print("title of Column : ",col,"-----",merge_df_5.shape[0]-merge_df_5[col].value_counts().sum()," NAN")
        merge_df_5[col].value_counts()
        print(merge_df_5[col].value_counts())
        print('\n')
        print('\n')
    print("✅ statistiques")

    # ----------------------------------------------------------------------------------------------------
    ### Etape 9 : Export de la table A+B
    return merge_df_5
