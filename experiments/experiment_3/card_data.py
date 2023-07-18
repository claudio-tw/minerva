import pandas as pd
import numpy as np
import category_encoders as ce


def transform_original_dataset(dataset: pd.DataFrame):
    num_samples = 500000
    random_state = 40
    df = dataset.sample(
        n=num_samples, random_state=random_state, ignore_index=True)
    num_of_features = df.shape[1] - 1
    num_of_targets = 1

    # 3. Convert counters from int to float, so that they are not considered categorical

    count_cols = [
        'account_dormancy_from_last_transfer',
        'ach_payments_cnt_1_week',
        'backward_route_pmnts_cnt',
        'balance1d_count',
        'balance7d_count',
        'bank1d_count',
        'cad_submitted_cnt_1_day',
        'card_at_least_1_fraud',
        'card_fraud_cnt',
        'card_max_fraudsters',
        'card_same_recipient_1d',
        'card_same_recipient_1h',
        'card_total_count',
        'card_total_fraudsters',
        'card_total_other_users',
        'card_users_cnt',
        'cash_advance_count_12h',
        'cash_advance_count_24h',
        'cash_advance_count_6h',
        'cash_advance_transaction_cnt',
        'cash_withdrawal_count_12h',
        'cash_withdrawal_count_24h',
        'cash_withdrawal_count_6h',
        'cash_withdrawal_transaction_cnt',
        'count_cards_1_day',
        'count_cards_1_week',
        'current_recipient_fraudster_count',
        'current_recipient_user_count',
        'dd_cad_same_recipient_1d',
        'dd_cad_same_recipient_1h',
        'dd_usd_same_recipient_1d',
        'dd_usd_same_recipient_1h',
        'device_at_least_1_fraud',
        'device_at_least_1_high_risk',
        'device_at_least_1_monitoring',
        'device_fonts_js_cnt',
        'device_max_fraudsters',
        'device_max_high_risk_users',
        'device_max_monitoring_users',
        'device_media_cnt',
        'device_total_count',
        'device_total_fraudsters',
        'device_total_high_risk',
        'device_total_monitoring',
        'device_total_other_users',
        'doc_matching_fraudster_count',
        'doc_matching_users_count',
        'ecom_purchase_count_12h',
        'ecom_purchase_count_24h',
        'ecom_purchase_count_6h',
        'ecom_purchase_transaction_cnt',
        'fraud_related_recipient_count',
        'fraudster_relation_count',
        'highest_fraudster_count',
        'highest_user_count',
        'ip_at_least_1_fraud',
        'ip_data_fraudulent_users_cnt',
        'ip_data_user_ip_subnet_cnt',
        'ip_data_users_cnt',
        'ip_max_fraudsters',
        'ip_total_count',
        'ip_total_fraudsters',
        'ip_total_other_users',
        'oldest_pmnt_age',
        'oldest_similar_recipient_pmnt_age',
        'oldest_user_similar_recipient_pmnt_age',
        'other_user_relation_count',
        'payin_account_age',
        'payin_account_user_age',
        'payin_attempt_count',
        'payment_cnt',
        'phone_and_email_and_password_change_14_days',
        'phone_and_email_and_password_change_30_days',
        'phone_and_password_change_14_days',
        'phone_and_password_change_30_days',
        'phone_fraud_cnt',
        'phone_users_cnt',
        'pos_purchase_count_12h',
        'pos_purchase_count_24h',
        'pos_purchase_count_6h',
        'pos_purchase_transaction_cnt',
        'recipient_bl_hits_count',
        'recipient_cnt',
        'recipient_count',
        'recipient_pmnt_cnt',
        'same_ccy_payments_cnt_12h',
        'same_ccy_payments_cnt_1d',
        'same_ccy_payments_cnt_3h',
        'same_route_age',
        'same_route_pmnts_cnt',
        'sender_age',
        'sender_completed_transfers',
        'sender_emailage_sm_friends',
        'significant_fraud_recipient_count',
        'source_ccy_cnt',
        'source_ccy_pmnt_cnt',
        'successful_bank_transfers_cnt',
        'target_ccy_cnt',
        'target_ccy_pmnt_cnt',
        'transfer_card_cnt',
        'transfer_submitted_afer_profile_creation_d',
        'transfer_submitted_after_profile_creation_log_s',
        'usd_submitted_cnt_1_day',
        'usd_submitted_cnt_1_week',
        'user_all_card_cnt',
        'user_card_refusal_count',
        'user_card_refusal_count7d',
        'user_review_count',
        'verif_acct_count',
        'verif_acct_failure',
        'verif_acct_holdernames',
        'verif_acct_users_blocked',
        'verif_acct_users_suspicious',
        'verifications_rejected_3months',
        'verifications_rejected_6months',
        'verifications_rejected_lifetime',
    ]

    df[count_cols] = df[count_cols].astype(float)

    assert num_of_features + num_of_targets == df.shape[1]

    # 4. Convert age to float, because it is ordinal

    age_cols = [
        'payin_account_age',
        'sender_age',
        'ip_data_age',
        'oldest_similar_recipient_pmnt_age',
        'payin_account_user_age',
        'same_route_age',
        'recipient_age',
        'oldest_user_similar_recipient_pmnt_age',
        'oldest_pmnt_age',
    ]

    df[age_cols] = df[age_cols].astype(float)

    assert num_of_features + num_of_targets == df.shape[1]

    # 5. Convert ratios and scores to float

    ratio_cols = [
        'transferflow_duration',
        'device_max_monitoring_ratio',
        'ip_max_ratio',
        'device_max_high_risk_ratio',
        'device_max_ratio',
        'recipient_emailage_score',
        'sender_emailage_score',
        'ip_data_period',
    ]

    df[ratio_cols] = df[ratio_cols].astype(float)

    assert num_of_features + num_of_targets == df.shape[1]

    # 6. Ordinal encoding of categorical features

    # Ordinal encoding of categorical features
    cattypes = ['object']
    catcols = list(df.select_dtypes(include=cattypes).columns)
    ordinal_encoder = ce.OrdinalEncoder(cols=catcols)
    ordinal_encoder.fit(df)
    df = ordinal_encoder.transform(df)

    assert num_of_features + num_of_targets == df.shape[1]

    # 7. Make all categorical variables start at 0

    cat_cols = df.select_dtypes(
        include=[int, np.int32, np.int64]).columns.tolist()
    baseline = df[cat_cols].min()
    assert not baseline.isna().any()
    df[cat_cols] -= baseline

    # 8. Rename features with muted names

    cat_cols = df.select_dtypes(
        include=[int, np.int32, np.int64, 'object']).columns.tolist()
    if 'label' in cat_cols:
        cat_cols.remove('label')
    float_cols = df.select_dtypes(
        include=[float, np.float32, np.float64]).columns.tolist()
    if 'label' in float_cols:
        float_cols.remove('label')
    targets = ['label']
    assert num_of_features + num_of_targets == df.shape[1]
    assert len(cat_cols) + len(float_cols) + len(targets) == df.shape[1]
    assert set(targets).intersection(
        set(float_cols).union(set(cat_cols))) == set()

    rename = {
        **{col: f'xc{n}' for n, col in enumerate(cat_cols)},
        **{col: f'xf{n}' for n, col in enumerate(float_cols)},
        **{'label': 'y'}
    }
    df.rename(columns=rename, inplace=True)
    df.sort_index(axis=1, inplace=True)
    return df


def main(
    uri='s3://ml-production-fraud-sagemaker-data/minerva_experiment/data/card/train.parquet/'
    destination='data/exp3.csv'
):
    dataset = pd.read_parquet(uri)
    df = transform_original_dataset(dataset)
    # 8. Repeat data so that the training sees equal amount of 0s and 1s
    idx_y1 = df['y'] == 1
    num1 = sum(idx_y1)
    num_repeat = len(df) // num1
    df1 = df.loc[idx_y1].copy()
    df = pd.concat([df] + ([df1]*num_repeat), axis=0, ignore_index=True)
    num_samples = 800000
    random_state = 40
    df = df.sample(num_samples, random_state=random_state, ignore_index=True)
    df.to_csv(destination, index=False)


if __name__ == '__main__':
    main()
