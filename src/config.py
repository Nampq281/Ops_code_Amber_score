#REQUIRED COLUMNS

df_root_col = ['id_customer2',
       'Contract.Instalments.Summary.NumberOfTerminated',
       'Contract.NonInstalments.Summary.NumberOfTerminated',
       'Contract.Cards.Summary.NumberOfRenounced',
       'Contract.Cards.Summary.NumberOfTerminated']
       

contract_level_noninstall_col = ['id_customer2', 'loan_code_lv2',
                                 'CommonData.TypeOfFinancing']

contract_level_install_col = ['id_customer2', 'loan_code_lv2',
                              'CommonData.ContractPhase',
                              'CommonData.TypeOfFinancing',
                              'RemainingInstalmentsAmount',
                              'TotalAmount',
                              'RemainingInstalmentsNumber',
                              'TotalNumberOfInstalments']

contract_level_card_col = ['id_customer2', 'loan_code_lv2', 'CreditLimit']


ts_col = ['loan_code_lv2', 'id_customer2', 'CommonData.CBContractCode',
          'ReferenceYear', 'ReferenceMonth', 
          'Default', 'Status',
          'ResidualAmount', 'Utilization', 'GuarantedAmount', 'Granted']
