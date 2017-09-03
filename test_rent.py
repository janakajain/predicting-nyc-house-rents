import homework2_rent
from homework2_rent import score_rent

def test_rent():
	R2 = score_rent()
	assert  R2> 0.5