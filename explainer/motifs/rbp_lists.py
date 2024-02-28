RBP_SUBSETS = {
    'encode': [
        'AATF', 'ADAR', 'AKAP8L', 'AQR', 'BUD13', 'CCAR1', 'CCAR2', 'CDC40',
        'CELF1', 'DAZAP1', 'DDX20', 'DDX42', 'DDX5', 'EFTUD2', 'EWSR1', 'FMR1',
        'FUBP1', 'FUS', 'GEMIN5', 'GPKOW', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPC',
        'HNRNPD', 'HNRNPF', 'HNRNPH1', 'HNRNPK', 'HNRNPL', 'HNRNPLL', 'HNRNPM',
        'HNRNPU', 'KHDRBS1', 'KHSRP', 'MATR3', 'NCBP2', 'NONO', 'PCBP1',
        'PCBP2', 'PPIG', 'PPP1R8', 'PRPF4', 'PRPF6', 'PRPF8', 'PSIP1', 'PTBP1',
        'PUF60', 'QKI', 'RAVER1', 'RBFOX2', 'RBM14', 'RBM15', 'RBM17', 'RBM22',
        'RBM25', 'RBM39', 'RBM47', 'RBM5', 'SAFB2', 'SART3', 'SF1', 'SF3A3',
        'SF3B1', 'SF3B4', 'SFPQ', 'SMN1', 'SMNDC1', 'SND1', 'SNRNP200',
        'SNRNP70', 'SRSF1', 'SRSF3', 'SRSF4', 'SRSF5', 'SRSF7', 'SRSF9',
        'STAU1', 'SUGP2', 'TARDBP', 'TFIP11', 'TIA1', 'TIAL1', 'TRA2A',
        'U2AF1', 'U2AF2', 'ZRANB2'
    ],

    'rosina2017': ['A1CF', 'ACO1', 'AKAP1', 'ANKHD1', 'CELF1', 'CELF4', 'CELF5',
                   'CELF6', 'CNOT4', 'CPEB2', 'CPEB4', 'CSTF2', 'DAZAP1', 'EIF4B',
                   'ELAVL1', 'ELAVL2', 'ELAVL3', 'ENOX1', 'ESRP2', 'FMR1', 'FUS',
                   'FXR1', 'FXR2', 'G3BP1', 'G3BP2', 'GRSF1', 'HNRNPA1', 'HNRNPA1L2',
                   'HNRNPA2B1', 'HNRNPA3', 'HNRNPAB', 'HNRNPC', 'HNRNPCL1', 'HNRNPF',
                   'HNRNPH1', 'HNRNPH2', 'HNRNPH3', 'HNRNPK', 'HNRNPL', 'HNRNPLL', 'HNRNPM',
                   'HNRNPU', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'KHDRBS1', 'KHDRBS2', 'KHDRBS3', 'KHSRP',
                   'LIN28A', 'MATR3', 'MBNL1', 'MSI1', 'NCL', 'NONO', 'PABPC1', 'PABPC3', 'PABPC5',
                   'PABPN1', 'PCBP1', 'PCBP2', 'PPRC1', 'PTBP1', 'PUM1', 'PUM2', 'QKI', 'RALY',
                   'RBFOX1', 'RBFOX2', 'RBM24', 'RBM28', 'RBM3', 'RBM4', 'RBM41', 'RBM42', 'RBM45',
                   'RBM46', 'RBM5', 'RBM6', 'RBM8A', 'RBMS1', 'RBMS3', 'RBMY1A1', 'SAMD4A', 'SART3',
                   'SF1', 'SFPQ', 'SNRNP70', 'SNRPA', 'SRSF1', 'SRSF10', 'SRSF2', 'SRSF3', 'SRSF5', 'SRSF6',
                   'SRSF7', 'SRSF9', 'SSB', 'TARDBP', 'TIA1', 'TRA2A', 'TRA2B', 'TUT1', 'U2AF2',
                   'YBX1', 'YBX2', 'YTHDC1', 'ZC3H10', 'ZC3H14', 'ZCRB1', 'ZFP36', 'ZFP36L2', 'ZNF638', 'ZRANB2'],

    'encode_in_rosina2017': ['SF1', 'QKI', 'TARDBP', 'SRSF3', 'TIA1', 'CELF1', 'HNRNPH1', 'SRSF1',
                             'NONO', 'RBM5', 'HNRNPU', 'SART3', 'KHSRP', 'ZRANB2', 'FMR1',
                             'SRSF5', 'PCBP1', 'MATR3', 'HNRNPF', 'SFPQ', 'PTBP1', 'TRA2A',
                             'PCBP2', 'DAZAP1', 'HNRNPK', 'HNRNPLL', 'RBFOX2', 'HNRNPA2B1',
                             'SRSF7', 'SRSF9', 'SNRNP70', 'HNRNPM', 'FUS', 'U2AF2',
                             'HNRNPL', 'KHDRBS1', 'HNRNPA1', 'HNRNPC'],
    
    'encode_in_attract': ['ADAR', 'CELF1', 'DAZAP1', 'FMR1', 'FUS', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPC', 'HNRNPD', 'HNRNPF',
                          'HNRNPH1', 'HNRNPK', 'HNRNPL', 'HNRNPLL', 'HNRNPM', 'HNRNPU', 'KHDRBS1', 'KHSRP', 'MATR3', 'NONO',
                          'PCBP1', 'PCBP2', 'PTBP1', 'QKI', 'RBFOX2', 'RBM14', 'RBM25', 'RBM5', 'SART3', 'SF1', 'SFPQ', 'SNRNP70',
                          'SRSF1', 'SRSF3', 'SRSF4', 'SRSF5', 'SRSF7', 'SRSF9', 'TARDBP', 'TIA1', 'TIAL1', 'TRA2A', 'U2AF2', 'ZRANB2']
    }