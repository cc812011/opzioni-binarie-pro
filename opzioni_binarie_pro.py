if model_type == "XGBoost (Veloce)":
            model = xgb.XGBClassifier(
                n_estimators=500, 
                max_depth=8, 
                learning_rate=0.04, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=42
            )
            model.fit(X_train, y_train)
            latest_features = data[feature_cols].iloc[-1:].values
            prob = model.predict_proba(latest_features)[0]
        else:
            # LSTM disattivato temporaneamente
            st.warning("LSTM non disponibile su Streamlit Cloud per limiti di risorse. Uso XGBoost.")
            model = xgb.XGBClassifier(...)  # stessa configurazione di sopra
            model.fit(X_train, y_train)
            latest_features = data[feature_cols].iloc[-1:].values
            prob = model.predict_proba(latest_features)[0]
