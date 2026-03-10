def get_sbs(row, genome, chr_id):
    ref = genome.fetch(chr_id['chr' + str(row['chr'])], row['start'] - 2, row['start'] + 1).upper()
    if('N' in ref):
        return None
    
    alt = row['alt']

    if(ref[1] != row['ref']):
        print('Reference mismatch:', row['chr'], row['start'], row['ref'], ref)

    if(ref[1] == 'A' or ref[1] == 'G'):
        ref = reverse_complement(ref)
        alt = reverse_complement(alt)

    return ref[0] + '[' + ref[1] + '>' + alt + ']' + ref[2]

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement[base] for base in reversed(seq))
