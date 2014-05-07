#!perl
use strict;
use warnings;

# For server
use HTTP::Daemon;
use HTTP::Status;
use HTTP::Response;
use JSON qw(encode_json decode_json);

use Data::Dumper;

# For similairty queries
use Digest::SHA1;
use WordNet::Similarity::lesk;
use WordNet::Similarity::vector;


my $http = HTTP::Daemon->new(
  LocalPort => 8080
  ) || die ("Cannot start server\n");

  print "Server is running on ", $http->url ,"\n";

  # Initialize wordnet for similarity queries
  my $wn = WordNet::QueryData->new();
  my $vector = WordNet::Similarity::vector->new($wn);
  my $lesk = WordNet::Similarity::lesk->new($wn, 'lesk.conf');

  while( my $client = $http->accept ) {
      while( my $request = $client->get_request() ) {
            if( $request->method eq "POST" and
                $request->uri->path eq "/api" and
                $request->header("User-Agent") eq "pyclient"
                ) {

              # params
              my %param = %{decode_json($request->content)};
              my $type = $param{"type"};
              my $syn1 = $param{"syn1"};
              my $syn2 = $param{"syn2"};

              my $json = '';
              # Get scores
              if ( $type eq "lesk" ) {
                my $lesk_rel = $lesk->getRelatedness($syn1, $syn2);
                $json = encode_json({
                    "lesk" => $lesk_rel,
                });
              } elsif ( $type eq "vector" ) {
                my $vector_rel = $vector->getRelatedness($syn1, $syn2);
                $json = encode_json({
                    "vector" => $vector_rel,
                });
              }

              $client->send_response(HTTP::Response->new(
                RC_OK,
                undef,
                [ ],
                # return the result as json
                $json
              ));
            }
            else {
                $client->send_error(RC_FORBIDDEN);
            }
        }
        $client->close;
        undef $client;
}
