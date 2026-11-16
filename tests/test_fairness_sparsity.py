def test_fairness_sparsity_import():
    import train_fairness_sparsity
    assert callable(train_fairness_sparsity.main)
