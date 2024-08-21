Chemin_Data_XLSX='../Data/Import_xls_ref.xlsx'

def Table_Mu_Cl_import_prec(fichier_Injure,fichier_Jeu,Chemin_Data_XLSX):

    ### merge de table 2

    #table des blessures
    fichier_blessure_binaire=fichier_Injure[['PlayerKey','DM_M1']]

    # Merge playlist_data with injury_data
    merged_df = pd.merge(fichier_Jeu, fichier_blessure_binaire,
                         on = 'PlayerKey',
                         how = 'left')
    merged_df=merged_df.rename(columns={'DM_M1': 'Injured'})
    merged_df.Injured=merged_df.Injured.fillna(0)

    ###

    #condition1 = fichier_Injure['PlayerKey'] = 47307
    condition2 = fichier_Injure['GameID'] == '47307-10'
    condition3 = fichier_Injure['BodyPart'] == 'Ankle'

    fichier_Injure_drop = fichier_Injure[~(condition2 & condition3)]

    fichier_Injure_drop_=fichier_Injure_drop.drop(columns='PlayKey')

    ###

    import numpy as np
    # Merge playlist_data with the rest of the data injure
    merged_df_2 = merged_df.merge(fichier_Injure_drop_,
                         on = ('PlayerKey','GameID'),
                         how = 'left')

    ### Feat Eng 1 : fillna

    columns_1=['BodyPart','Surface','DM_M1','DM_M7','DM_M28','DM_M42','Weather','PlayType']

    for col in columns_1:
        merged_df_2[col]=merged_df_2[col].fillna(0)

    merged_df_2['DM_TOT']=merged_df_2['DM_M1']+merged_df_2['DM_M7']+merged_df_2['DM_M28']+merged_df_2['DM_M42']

    ### Feat Eng 2 : DM_GR

    merged_df_2['DM_GR'] = merged_df_2['DM_TOT'].apply(lambda x: 'GN' if x <= 2 else 'GO')

    merged_df_2['Temperature'] = merged_df_2['Temperature'].apply(lambda x: 0 if x <= -5 else x)

    merged_df_2['Temperature_C'] = merged_df_2['Temperature'].apply(lambda x: round((x-32)*5/9,0))

    ### Feat Eng 3 : replace nan

    merged_df_2['StadiumType']=merged_df_2['StadiumType'].replace(np.nan,'StadiumTypeNan')
    merged_df_2['PlayType']=merged_df_2['PlayType'].replace('Missing Data','PositionNan')
    merged_df_2['PlayType']=merged_df_2['PlayType'].replace('0','PositionNan')
    merged_df_2['Position']=merged_df_2['Position'].replace('Missing Data','PositionNan')
    merged_df_2['PositionGroup']=merged_df_2['PositionGroup'].replace('Missing Data','PositionGroupNan')

    ### Import des tables ref

    StadiumType_sheet=pd.read_excel(Chemin_Data_XLSX,sheet_name='StadiumType')

    Weather_sheet=pd.read_excel(Chemin_Data_XLSX,sheet_name='Weather')

    ### merge des tables pour rendre plus lisibles les données Weather et StadiumType

    merged_df_2['StadiumType']=merged_df_2['StadiumType'].apply(lambda x: x.strip() if isinstance(x, str) else x)

    merged_df_3 = merged_df_2.merge(StadiumType_sheet,
                         on = 'StadiumType',
                         how = 'left')

    merged_df_4 = merged_df_3.merge(Weather_sheet,
                         on = 'Weather',
                         how = 'left')

    ### Drop les colonnes suivantes

    Drop_Colonne_sheet=pd.read_excel(Chemin_Data_XLSX,sheet_name='DropSheet')

    drop_table=Drop_Colonne_sheet[Drop_Colonne_sheet.DROP_INDICATEUR.str.contains("OUI")]
    colonnes_a_droper=list(drop_table['DROP_COLONNE'])

    merge_df_5=merged_df_4.drop(colonnes_a_droper,axis=1)

    merge_df_5['Weather_Visibilite']=merge_df_5.Weather_Visibilite.isna().fillna('neutral')

    merge_df_5.describe(include='all').T

    print (merge_df_5.describe(include='all').T)
    print("✅ description")

    ### Export de la table A+B

    colonnes_du_data=list(merge_df_5.columns)

    for col in colonnes_du_data:
        print("title of Column : ",col,"-----",merge_df_5.shape[0]-merge_df_5[col].value_counts().sum()," NAN")
        merge_df_5[col].value_counts()
        print(merge_df_5[col].value_counts())
        print('\n')
        print('\n')

    print("✅ statistiques")

    return merge_df_5
