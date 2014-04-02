use WordNet::Similarity::lesk;
$wn = WordNet::QueryData->new();
$lesk = WordNet::Similarity::lesk->new($wn);
$relatedness = $lesk->getRelatedness($ARGV[0], $ARGV[1]);
print $relatedness;
