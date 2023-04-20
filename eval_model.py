for idx, (traindf, validationdf, testdf) in enumerate(
            generate_data_splits(data, strategy=strategy,
                                 periods_to_test=N_YEARS)):

    # x_train, y_train = traindf['text'], traindf['label']
    # x_test, y_test = testdf['text'], testdf['label']
    
    for k, (train_index, test_index) in enumerate(KFold(n_splits=K, shuffle=True).split(traindf)):
        traindf_k = traindf.iloc[train_index]

        """
        total_test = len(y_test)
        total_train = len(y_train)
        for idx, label in enumerate(['negative', 'positive', 'neutral']):
            prop_train = sum(y_train == idx)/total_train
            prop_test = sum(y_test == idx)/total_test
            print(f"{label} - test: {prop_test:0.4f}, train: {prop_train:0.4f}")
        """

        out, metrics = model(traindf_k, validationdf, testdf)
        out['Date'] = pd.to_datetime(out['Date'], format="%Y%m%d")
        metrics.year = testdf.Date.iloc[0].year
        metrics.k = k + 1
        print(f'{metrics.year}, k = {k+1}')
        print(metrics)
        all_metrics.append(metrics)

        # get_reference_data(out, yd, cols=['Annual Return', 'beta', 'sp Annual', 'sp Percent'])

        temp_df = pd.DataFrame()
        temp_df[['tikr', 'Date','label', 'pred', 'score']] = out[['tikr', 'Date','label', 'pred', 'score']]

        temp_df['pred'] = temp_df['pred'].astype(int)
        temp_df['year'] = [metrics.year]*len(out)
        temp_df['k'] = [k+1]*len(out)
        output_df = pd.concat([output_df, temp_df])

if not os.path.exists('figs'):
    os.makedirs('model_outputs')
output_df.reset_index(drop=True).to_csv(f'model_outputs/outputs_{model_savepath}.csv')