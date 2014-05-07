use local::lib;
use Digest::SHA1;
use WordNet::Similarity::lesk;
use WordNet::Similarity::vector;

$wn = WordNet::QueryData->new();
$vector = WordNet::Similarity::vector->new($wn);
$lesk = WordNet::Similarity::lesk->new($wn, '/u/leila/sensim/lesk.conf');

$lesk_rel = $lesk->getRelatedness($ARGV[0], $ARGV[1]);
$vector_rel = $vector->getRelatedness($ARGV[0], $ARGV[1]);

print $lesk_rel;
print $vector_rel;
