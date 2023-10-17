function [atoms, lx, ly, lz, a, b, c, dz] = Diamond110_xtl(na, nb, nc, ncu, rmsd_3d)
    xtl_parm.na = na;
    xtl_parm.nb = nb;
    xtl_parm.nc = nc;
    
    xtl_parm.a = 3.56/sqrt(2);
    xtl_parm.b = 3.56;
    xtl_parm.c = sqrt(2)*3.56/2;
    
    xtl_parm.alpha = 90;
    xtl_parm.beta = 90;
    xtl_parm.gamma = 90;
    
    xtl_parm.sgn = 1;
    xtl_parm.pbc = false;
    xtl_parm.asym_uc = [];
    
    occ = 1;
    tag = 0;
    charge = 0;
    
    % C = 6
    % Z x y z rmsd_3d occupancy charge 
    xtl_parm.base = [6, 0.00, 0.00, 0.00, rmsd_3d, occ, tag, charge;...
                        6, 0.50, 0.75, 0.00, rmsd_3d, occ, tag, charge;...
                        6, 0.00, 0.25, 0.50, rmsd_3d, occ, tag, charge;...
                        6, 0.50, 0.50, 0.50, rmsd_3d, occ, tag, charge];
                    
    atoms = ilc_xtl_build(xtl_parm);

    dz = xtl_parm.c/ncu;
    lx = na*xtl_parm.a;ly = nb*xtl_parm.b;lz = nc*xtl_parm.c;
    
    a = xtl_parm.a;
    b = xtl_parm.b;
    c = xtl_parm.c;
end