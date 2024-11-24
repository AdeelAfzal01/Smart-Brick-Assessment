FILLAFTERSORTING_COLUMNS = ["is_freehold", "nearest_landmark_en", "nearest_mall_en", "nearest_metro_en"]
SORTING_COLUMN = 'area_en'
FILLWITHUNKNOWN_COLUMNS = ['property_subtype_en', 'rooms_en']
ONEHOTENCODER_COLUMNS = ['transaction_type_en', 'registration_type_en', 'is_freehold_text', 'property_usage_en', 'is_offplan','is_freehold', 'property_type_en', 'nearest_mall_en']
LABELENCODER_COLUMNS = ['nearest_landmark_en', 'transaction_subtype_en', 'property_subtype_en', 'area_en', 'nearest_metro_en']
ROOMSEN_COLUMN = 'rooms_en'
TRANSACTIONDATETIME_COLUMN = "transaction_datetime"
REMOVEOUTLIER_COLUMNS = ["amount", "transaction_age_in_days"]
SCALINGFEATURES_COLUMNS = LABELENCODER_COLUMNS + ['rooms_en', 'transaction_age_in_days'] 
TARGET_COLUMN = "amount"

