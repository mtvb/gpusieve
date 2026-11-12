#!/usr/bin/perl -p
# Rewrites CADO-format relations (e.g. from NFS_factory) the way las prints them: lowercase hex, primes of each side in
# ascending order. Then the same relation from las and from gpusieve + NFS_factory is the same line, so sort -u removes it.
#     lasnorm.pl rels.txt > rels.las.txt        (or as a filter in a pipe)
if (!/^#/) {
	chomp;
	my ($ab, @sides) = split /:/, $_, -1;
	$_ = join(":", $ab, map { join(",", map { sprintf "%x", $_ } sort { $a <=> $b } map { hex } split /,/) } @sides) . "\n";
}
