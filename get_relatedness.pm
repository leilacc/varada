use Digest::SHA1;
use WordNet::Similarity::lesk;
use WordNet::Similarity::vector;

$wn = WordNet::QueryData->new();
$vector = WordNet::Similarity::vector->new($wn);
$lesk = WordNet::Similarity::lesk->new($wn, 'lesk.conf');

$syn1 = <>;
$syn2 = <>;

$lesk_rel = $lesk->getRelatedness($syn1, $syn2);
$vector_rel = $vector->getRelatedness($syn1, $syn2);

while (1) {
print $lesk_rel;
print $vector_rel;
}
