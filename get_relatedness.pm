use WordNet::Similarity::lesk;
$wn = WordNet::QueryData->new();
$lesk = WordNet::Similarity::lesk->new($wn);
print 'ok';

while (1) {
  #$w1 = <>;
  #$w2 = <>;
  $w1 = "cat#n#1";
  $w2 = "cat#n#1";
  $relatedness = $lesk->getRelatedness($w1, $w2);
  print $relatedness;
}
