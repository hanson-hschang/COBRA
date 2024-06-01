from cobr2.actuations import ContinuousActuation


def test_actuation_shape():

    n_dim = 3
    n_elements = 10
    actuation = ContinuousActuation(n_elements=n_elements)

    assert actuation.internal_force.shape == (n_dim, n_elements)
    assert actuation.internal_couple.shape == (n_dim, n_elements - 1)
    assert actuation.equivalent_external_force.shape == (n_dim, n_elements + 1)
    assert actuation.equivalent_external_couple.shape == (n_dim, n_elements)


def test_actuation_reset():

    n_elements = 10
    actuation = ContinuousActuation(n_elements=n_elements)

    actuation.internal_force[0, :] = 1
    actuation.internal_couple[0, :] = 1
    actuation.equivalent_external_force[0, :] = 1
    actuation.equivalent_external_couple[0, :] = 1

    actuation.reset_actuation()

    assert (actuation.internal_force == 0).all()
    assert (actuation.internal_couple == 0).all()
    assert (actuation.equivalent_external_force == 0).all()
    assert (actuation.equivalent_external_couple == 0).all()
