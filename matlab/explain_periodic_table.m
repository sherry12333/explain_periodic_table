clc,clear,close all

% K

% start with V_ee =0

% hbh_matrix is different for different l

% n=4
Gauss4=[-sqrt(3/7-2/7*sqrt(6/5)),(18+sqrt(30))/36;sqrt(3/7-2/7*sqrt(6/5)),(18+sqrt(30))/36;
    -sqrt(3/7+2/7*sqrt(6/5)),(18-sqrt(30))/36;sqrt(3/7+2/7*sqrt(6/5)),(18-sqrt(30))/36];

Gauss_mat=Gauss4;

% parameters

a0=1; % m
H=1; %  Hartree
parameter1=0.5*H*a0^2; % h_bar^2/(2*m)
parameter2=2*parameter1/a0; % e^2/(4*pi*epsilon0)

epsilon0=1/(4*pi);
charge=1;

kord=4;
h=0.1; 
h1=0.5;
xmin=0;xmax=10;

% linear knot sequence
xarray_P1=xmin:h:5;
xarray_P2=5+h1:h1:xmax;
xarray_P=[xarray_P1,xarray_P2];

xarray_V=xarray_P;

x_ghost_former=xmin*ones(1,kord-1);
x_ghost_behind=xmax*ones(1,kord-1);
tknot_P=[x_ghost_former,xarray_P,x_ghost_behind];

tknot_V=tknot_P;

eta=0.4;

l0=0;
l1=1;
l2=0;

Z=19; % He

% occupation number
N10=2; % 1s
N20=2; % 2s
N21=6; % 2p

N30=2; % 3s
N31=6; % 3p
N32=[1,0];
N40=0;



iteration=30;
tic
[E_tot,ryo_K]=cal_ryo_E(eta,iteration,charge,epsilon0,N10,N20,N21,N30,N31,N32,N40(1,1),xarray_P,xarray_V,tknot_P,tknot_V,kord,Gauss_mat,parameter1,parameter2,l0,l1,l2,Z);
toc

[E_tot_ion,ryo_K_ion]=cal_ryo_E(eta,iteration,charge,epsilon0,N10,N20,N21,N30,N31,N32,N40(1,2),xarray_P,xarray_V,tknot_P,tknot_V,kord,Gauss_mat,parameter1,parameter2,l0,l1,l2,Z);

figure()
subplot(2,2,1)
plot(xarray_P,ryo_K.*(4*pi*xarray_P.^2),'LineStyle','--','LineWidth',2);hold on

plot(xarray_P,ryo_K_ion.*(4*pi*xarray_P.^2),'LineStyle',':','LineWidth',2);



xlabel('r [a_0]')
ylabel('ρ(r) × 4πr^2')
title('(b)')
legend('3p_63d_1','3p_63d_1+')

% subplot(2,2,2)
% plot(xarray_P,ryo_ar,'LineStyle','--','LineWidth',2);hold on
% 
% plot(xarray_P,ryo_ar_ion,'LineStyle',':','LineWidth',2);
% 
% xlabel('r [a_0]')
% ylabel('ρ(r)')
% title('(b)')
% legend('Ar','Ar+')

% check if integral of ryo is N_occ*e
[integral]=cal_integral_ryo(ryo_K(1,2:end),xarray_P(1,2:end));
N=integral/charge;

[integral]=cal_integral_ryo(ryo_K_ion(1,2:end),xarray_P(1,2:end));
N_ion=integral/charge;





function [E_tot,ryo_ne]=cal_ryo_E(eta,iteration,charge,epsilon0,N10,N20,N21,N30,N31,N32,N40,xarray_P,xarray_V,tknot_P,tknot_V,kord,Gauss_mat,parameter1,parameter2,l0,l1,l2,Z)
% hydrogen-like solution

[c0,D0,bhb_matrix0,BB_matrix0]=cal_eigen(xarray_P,tknot_P,kord,Gauss_mat,parameter1,parameter2,l0,Z);
[c1,D1,bhb_matrix1,BB_matrix1]=cal_eigen(xarray_P,tknot_P,kord,Gauss_mat,parameter1,parameter2,l1,Z);
[c2,D2,bhb_matrix2,BB_matrix1]=cal_eigen(xarray_P,tknot_P,kord,Gauss_mat,parameter1,parameter2,l2,Z);


E_hyd1=D0(1,1);E_hyd2=D0(2,2); E_hyd3=D0(3,3);


[Bavx1,dBavx1,dB2avx1] = bsplgen(xarray_P,tknot_P,kord);

P10=Bavx1(:,2:end-1)*c0(:,1); %

P20=Bavx1(:,2:end-1)*c0(:,2); %
P21=Bavx1(:,2:end-1)*c1(:,1); %

P30=Bavx1(:,2:end-1)*c0(:,3); 
P31=Bavx1(:,2:end-1)*c1(:,2); 
P32=Bavx1(:,2:end-1)*c2(:,1); 

P40=Bavx1(:,2:end-1)*c0(:,4); 

ryo_ne=1/(4*pi)*charge*(N10*(P10'./xarray_P).^2+N20*(P20'./xarray_P).^2+N21*(P21'./xarray_P).^2+N30*(P30'./xarray_P).^2+N31*(P31'./xarray_P).^2+N32*(P32'./xarray_P).^2+N40*(P40'./xarray_P).^2);

RHS_ne=-xarray_P.*ryo_ne*4*pi/(4*pi*epsilon0);
RHS_ne(1,1)=RHS_ne(1,2);

[dB2_matrix]=build_matrix_dB2(xarray_P,dB2avx1,dBavx1);


[c]=l_u(dB2_matrix,RHS_ne');
dB_end_end_1=dBavx1(end,end-1);
dB_end=dBavx1(end,end);
c_end_1=c(end,1);

c_end=(0-dB_end_end_1)/dB_end*c_end_1;

c=[0;c;c_end];


% electron potential is considered

[bhb_V_ee_matrix_old]=cal_V_ee_matrix(c,bhb_matrix0,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,Gauss_mat,parameter1,parameter2,l0,Z);


V_old=bhb_V_ee_matrix_old;

eigen10=zeros(1,iteration);eigen20=zeros(1,iteration);eigen21=zeros(1,iteration);eigen30=zeros(1,iteration);eigen31=zeros(1,iteration);eigen32=zeros(1,iteration);eigen40=zeros(1,iteration);

for i=1:iteration
[c0,D0]=cal_eigen1(bhb_V_ee_matrix_old,bhb_matrix0,BB_matrix0);
[c1,D1]=cal_eigen1(bhb_V_ee_matrix_old,bhb_matrix1,BB_matrix0);
[c2,D2]=cal_eigen1(bhb_V_ee_matrix_old,bhb_matrix2,BB_matrix0);


P10=Bavx1(:,2:end-1)*c0(:,1); %

P20=Bavx1(:,2:end-1)*c0(:,2); %
P21=Bavx1(:,2:end-1)*c1(:,1); %

P30=Bavx1(:,2:end-1)*c0(:,3); 
P31=Bavx1(:,2:end-1)*c1(:,2); 
P32=Bavx1(:,2:end-1)*c2(:,1); 

P40=Bavx1(:,2:end-1)*c0(:,4); 

ryo_ne=1/(4*pi)*charge*(N10*(P10'./xarray_P).^2+N20*(P20'./xarray_P).^2+N21*(P21'./xarray_P).^2+N30*(P30'./xarray_P).^2+N31*(P31'./xarray_P).^2+N32*(P32'./xarray_P).^2+N40*(P40'./xarray_P).^2);

RHS_ne=-xarray_P.*ryo_ne*4*pi/(4*pi*epsilon0);
RHS_ne(1,1)=RHS_ne(1,2);

[dB2_matrix]=build_matrix_dB2(xarray_P,dB2avx1,dBavx1);

[c]=l_u(dB2_matrix,RHS_ne');
dB_end_end_1=dBavx1(end,end-1);
dB_end=dBavx1(end,end);
c_end_1=c(end,1);

c_end=(0-dB_end_end_1)/dB_end*c_end_1;

c=[0;c;c_end];

eigen10(1,i)=D0(1,1);eigen20(1,i)=D0(2,2);eigen21(1,i)=D1(1,1);eigen30(1,i)=D0(3,3);eigen31(1,i)=D1(2,2);eigen32(1,i)=D2(1,1);eigen40(1,i)=D0(4,4);
[bhb_V_ee_matrix_new]=cal_V_ee_matrix(c,bhb_matrix0,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,Gauss_mat,parameter1,parameter2,l0,Z);

V_new=bhb_V_ee_matrix_new;

V_new=V_new*(1-eta)+V_old*eta;

V_old=V_new;

bhb_V_ee_matrix_old=V_new;
end


ryo_ne(1,1)=0;

size_c=size(c0);

E_orb10=eigen10(1,end);E_orb20=eigen20(1,end);E_orb21=eigen21(1,end);E_orb30=eigen30(1,end);E_orb31=eigen31(1,end);E_orb32=eigen32(1,end);E_orb40=eigen40(1,end);


sum10=0;sum20=0;sum21=0;sum30=0;sum31=0;sum32=0;sum40=0;
for i=1:size_c(1,1)
    sum10=sum10+bhb_V_ee_matrix_new(i,:)*(c0(i,1)*c0(:,1));
    sum20=sum20+bhb_V_ee_matrix_new(i,:)*(c0(i,2)*c0(:,2));
    sum21=sum21+bhb_V_ee_matrix_new(i,:)*(c1(i,1)*c1(:,1));
    sum30=sum30+bhb_V_ee_matrix_new(i,:)*(c0(i,3)*c0(:,3));
    sum31=sum31+bhb_V_ee_matrix_new(i,:)*(c1(i,2)*c1(:,2));
    sum32=sum32+bhb_V_ee_matrix_new(i,:)*(c2(i,1)*c2(:,1));
    sum40=sum40+bhb_V_ee_matrix_new(i,:)*(c0(i,4)*c0(:,4));


end


E_tot10=N10*(E_orb10-sum10/2);
E_tot20=N20*(E_orb20-sum20/2);
E_tot21=N21*(E_orb21-sum21/2);
E_tot30=N30*(E_orb30-sum30/2);
E_tot31=N31*(E_orb31-sum31/2);
E_tot32=N32*(E_orb32-sum32/2);
E_tot40=N40*(E_orb40-sum40/2);

E_tot=E_tot10+E_tot20+E_tot21+E_tot30+E_tot31+E_tot32+E_tot40;
end






% calculate potential V_ee_dir at transform_coor
function [Pot,P_V]=cal_potential(transform_coor,xintervall,tknot,kord,RHS)
[Bavx,dBavx,dB2avx] = bsplgen(xintervall,tknot,kord);
[dB2_matrix]=build_matrix_dB2(xintervall,dB2avx,dBavx);

[Bavx1,dBavx1,dB2avx1] = bsplgen([transform_coor,transform_coor+1],tknot,kord);
% [B_matrix]=build_matrix_B([transform_coor,transform_coor+1],Bavx1);
B_Gaussian=Bavx1(1,:);
% [B_matrix]=build_matrix_B(xintervall,Bavx);

[c]=l_u(dB2_matrix,RHS);
dB_end_end_1=dBavx(end,end-1);
dB_end=dBavx(end,end);
c_end_1=c(end,1);

c_end=(0-dB_end_end_1)/dB_end*c_end_1;
c0=0;
c=[c0;c;c_end];

result=B_Gaussian*c;

Pot=result/transform_coor;
% Pot(1,1)=Pot(1,2);
end


% build matrix dB2
function [matrix]=build_matrix_dB2(xintervall,dB2,dB1)
matrix=zeros(length(xintervall),length(xintervall));
for i=1:length(xintervall)-1
    matrix(i,i)=dB2(i,i+1);
    matrix(i,i+1)=dB2(i,i+2);
    matrix(i+1,i)=dB2(i+1,i+1);
end
matrix(end,end)=dB2(end,end-1)-dB1(end,end-1)*dB2(end,end)/dB1(end,end);
end

% build matrix B
function [matrix]=build_matrix_B(xintervall,dB2)
matrix=zeros(length(xintervall),length(xintervall));
for i=1:length(xintervall)-1
    matrix(i,i)=dB2(i,i+1);
    matrix(i,i+1)=dB2(i,i+2);
    matrix(i+1,i)=dB2(i+1,i+1);
end
matrix(end,end)=dB2(end,end-1);
end

% LU factorization
function [X]=l_u(A,B)
A=sparse(A);
[L,U]=lu(A);
Y=L\B;
X=U\Y;
end


% insert extra x coordinate into array, the added x coordinate is not equal to items in
% original xarray
function [xarray_insert]=insert_array(x,xarray)
index=0;
% x_in_array=find(xarray==x);
for i=1:length(xarray)-1
    if (x>xarray(1,i))*(x<xarray(1,i+1))
        index=i;

    end
end
xarray_insert=[xarray(1,1:index),x,xarray(1,index+1:end)];

end


% calculate P=rR
function [V,D,bhb_matrix,BB_matrix]=cal_eigen(xarray,tknot,kord,Gauss_mat,parameter1,parameter2,l,Z)
% matrix of B
BB_matrix=zeros(length(tknot)-kord-2,length(tknot)-kord-2);
for j=2:length(tknot)-kord-1
    for i=2:length(tknot)-kord-1
        [BB_matrix(j-1,i-1)]=cal_inte_BBinterval(xarray,tknot,kord,i,j,Gauss_mat);
    end
end

% matrix of H
bhb_matrix=zeros(length(tknot)-kord-2,length(tknot)-kord-2);
for j=2:length(tknot)-kord-1
    for i=2:length(tknot)-kord-1
        [bhb_matrix(j-1,i-1)]=cal_bhb_interval(xarray,tknot,kord,i,j,parameter1,parameter2,Gauss_mat,l,Z);
    end
end

[V,D]=eig(bhb_matrix,BB_matrix);
end


% calculate P=rR
% function [V,D]=cal_eigen1(bhb_V_ee_matrix,bhb_matrix,BB_matrix,c,N10,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,Gauss_mat,parameter1,parameter2,l,Z)

function [V,D]=cal_eigen1(bhb_V_ee_matrix,bhb_matrix,BB_matrix)

bhb_matrix1=bhb_V_ee_matrix+bhb_matrix;

[V,D]=eig(bhb_matrix1,BB_matrix);
end

function [bhb_V_ee_matrix]=cal_V_ee_matrix(c,bhb_matrix,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,Gauss_mat,parameter1,parameter2,l,Z)
bhb_V_ee_matrix=zeros(length(tknot_P)-kord-2,length(tknot_P)-kord-2);
for j=2:length(tknot_P)-kord-1
    for i=2:length(tknot_P)-kord-1
        if (bhb_matrix(j-1,i-1)~=0)
        [bhb_V_ee_matrix(j-1,i-1)]=cal_bhb_interval1(c,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,i,j,Gauss_mat,parameter1,parameter2,l,Z);
        end
    end
end
end
 

% calculate integral of B_iB_j in some interval
function [BBinterval]=cal_inte_BBinterval(xarray,tknot,kord,index1,index2,Gauss_mat)
sum=0;
for i=1:length(xarray)-1
a=xarray(1,i);b=xarray(1,i+1);
    [BB]=cal_integral(tknot,kord,index1,index2,@fun1,Gauss_mat,a,b,1);
    sum=sum+BB;
end
BBinterval=sum;
end

% calculate integral of B_iB_j in some interval
function [BBinterval]=cal_inte_BBinterval1(c,N10,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,index1,index2,Gauss_mat)
sum=0;
for i=1:length(xarray_P)-1
a=xarray_P(1,i);b=xarray_P(1,i+1);
    [BB]=cal_integral1(c,N10,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,index1,index2,@fun1,Gauss_mat,a,b,1);
    sum=sum+BB;
end
BBinterval=sum;
end




% calculate integral of B_iHB_j in some interval
function [bhb_interval]=cal_bhb_interval(xarray,tknot,kord,index1,index2,parameter1,parameter2,Gauss_mat,l,Z)
sum=0;
for i=1:length(xarray)-1
    a=xarray(1,i);b=xarray(1,i+1);
    [bhb]=cal_BHB(tknot,kord,index1,index2,parameter1,parameter2,a,b,Gauss_mat,l,Z);
    sum=sum+bhb;
end
bhb_interval=sum;
end

% calculate integral of B_iHB_j in some interval
function [bhb_interval]=cal_bhb_interval1(c,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,index1,index2,Gauss_mat,parameter1,parameter2,l,Z)
sum=0;
for i=1:length(xarray_P)-1
    a=xarray_P(1,i);b=xarray_P(1,i+1);
    [bhb]=cal_BHB1(c,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,index1,index2,Gauss_mat,a,b,parameter1,parameter2,l,Z);
    sum=sum+bhb;
end
bhb_interval=sum;
end

% check integral of ryo
function [integral]=cal_integral_ryo(ryo,xarray)
sum=0;
for i=1:length(xarray)-1
    a=xarray(1,i);b=xarray(1,i+1);      
    sum=sum+ryo(1,i)*a^2*(b-a);

end
integral=sum*4*pi;
end



% Bi*Bj*fun
% Gauss_mat: Gauss with number n
% a: smallest limit of integral
% b: largest limit of integral
function [integral]=cal_integral(tknot,kord,index1,index2,fun,Gauss_mat,a,b,choose)
sum=0;N=size(Gauss_mat);
for i=1:N(1,1)
transform_coor=Gauss_mat(i,1)*(b-a)/2+(b+a)/2;
[Bavx,dBavx,dB2avx] = bsplgen([transform_coor,transform_coor+1],tknot,kord); % the value is correct only when input is array
if choose==1
    sum=sum+Bavx(1,index1)*Bavx(1,index2)*fun(transform_coor)*Gauss_mat(i,2);
else
    sum=sum+dBavx(1,index1)*dBavx(1,index2)*fun(transform_coor)*Gauss_mat(i,2);
end
end
integral=(b-a)/2*sum;
end


function [V_integral]=cal_integral1(c,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,index1,index2,fun,Gauss_mat,a,b,choose)
sum=0;N=size(Gauss_mat);


for i=1:N(1,1)
transform_coor=Gauss_mat(i,1)*(b-a)/2+(b+a)/2;


[Bavx,dBavx,dB2avx] = bsplgen([transform_coor,transform_coor+1],tknot_P,kord);

B_Gaussian=Bavx(1,:);
result=B_Gaussian*c;
Pot=result/transform_coor;
V_ee_dir=Pot;


P10_Gaussian=Bavx(1,2:end-1)*c0(:,1);
P20_Gaussian=Bavx(1,2:end-1)*c0(:,2);
P21_Gaussian=Bavx(1,2:end-1)*c1(:,1);

P30_Gaussian=Bavx(1,2:end-1)*c0(:,3); 
P31_Gaussian=Bavx(1,2:end-1)*c1(:,2); 
P32_Gaussian=Bavx(1,2:end-1)*c2(:,1); 

P40_Gaussian=Bavx(1,2:end-1)*c0(:,4); 

ryo_ne_Gaussian=1/(4*pi)*charge*(N10*(P10_Gaussian/transform_coor).^2+N20*(P20_Gaussian/transform_coor).^2+N21*(P21_Gaussian/transform_coor).^2+N30*(P30_Gaussian/transform_coor).^2+N31*(P31_Gaussian/transform_coor).^2+N32*(P32_Gaussian/transform_coor).^2+N40*(P40_Gaussian/transform_coor).^2);


V_ee_exch=-3*charge/(4*pi*epsilon0)*(3*ryo_ne_Gaussian/(charge*8*pi)).^(1/3);

V_ee=V_ee_dir+V_ee_exch;

if choose==1
    sum=sum+Bavx(1,index1)*Bavx(1,index2)*fun(transform_coor)*Gauss_mat(i,2);
else if (choose==2)
    sum=sum+dBavx(1,index1)*dBavx(1,index2)*fun(transform_coor)*Gauss_mat(i,2);
else
    sum=sum+Bavx(1,index1)*Bavx(1,index2)*V_ee*Gauss_mat(i,2);
end
end
end
V_integral=(b-a)/2*sum;
end


function [x2]=fun1(x)
x2=1;
end

function [x2]=fun2(x)
x2=1/x^2;
end

function [x2]=fun3(x)
x2=1/x;
end



% bhb include V_ee
function [bhb]=cal_BHB1(c,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,index1,index2,Gauss_mat,a,b,parameter1,parameter2,l,Z)
item4=charge*cal_integral1(c,c0,c1,c2,N10,N20,N21,N30,N31,N32,N40,charge,epsilon0,xarray_P,xarray_V,tknot_V,tknot_P,kord,index1,index2,@fun1,Gauss_mat,a,b,3);

bhb=item4;

end


function [bhb]=cal_BHB(tknot,kord,index1,index2,parameter1,parameter2,a,b,Gauss_mat,l,Z)
item1=parameter1*cal_integral(tknot,kord,index1,index2,@fun1,Gauss_mat,a,b,2);
item2=parameter1*l*(l+1)*cal_integral(tknot,kord,index1,index2,@fun2,Gauss_mat,a,b,1);
item3=-Z*parameter2*cal_integral(tknot,kord,index1,index2,@fun3,Gauss_mat,a,b,1);
bhb=item1+item2+item3;
end

% row of Bavx: the value of Bi at a knotpoint
% column of Bavx: basis function
function [Bavx,dBavx,dB2avx] = bsplgen(xintervall,tknot,kord)



istart=1;
islut=length(tknot)-kord;


punkter = length(xintervall);
Bavx=zeros(punkter,islut);
dBavx = zeros(punkter,islut);
dB2avx = zeros(punkter,islut);
    
xnr=0;

for x = xintervall
    
    B = zeros(length(tknot),kord);
    
    xnr=xnr+1;
    
    for k=1:kord
	
	
	
	for i=istart:islut    
	    
	    
	    
	    if k==1
		if tknot(i)< x & x < tknot(i+1)
		    
		    B(i,1)=1;
		    
		    
		elseif tknot(i) == x & x < tknot(i+1)
		    B(i,1)=1;
		    
		else
		    B(i,1)=0;
		end
		
	    elseif k>1
		
                
    

		
                if i < kord-k+1  
		
		
		    B(i,k)=0;
                    
 
		elseif B(i,k-1)==0 & tknot(i+k-1)-tknot(i)==0  
		    
		    if B(i+1,k-1) == 0
			B(i,k) =0;
			
		    else
			B(i,k)=(tknot(i+k)-x)/(tknot(i+k)-tknot(i+1))*B(i+1,k-1);
			

                        if k==kord 
		           dBavx(xnr,i) = (k-1)*( 0 - B(i+1,k-1)/(tknot(i+k)-tknot(i+1)));
                        end
                   
		    end
			
		    
		elseif  B(i+1,k-1)==0 & tknot(i+k)-tknot(i+1)==0  
		
		    B(i,k)=(x-tknot(i))/(tknot(i+k-1)-tknot(i))* ...
			   B(i,k-1);
		    
                    if k==kord 
		        dBavx(xnr,i) = (k-1)*(B(i,k-1)/(tknot(i+k-1)-tknot(i))-0);
		    end
                   
		else
		    B(i,k)=(x-tknot(i))/(tknot(i+k-1)-tknot(i))*B(i,k-1)+...
			   (tknot(i+k)-x)/(tknot(i+k)-tknot(i+1))*B(i+1,k-1);
		    
                    if k==kord  
		         dBavx(xnr,i) = (k-1)*(B(i,k-1)/(tknot(i+k-1)-tknot(i))-...
					      B(i+1,k-1)/(tknot(i+k)-tknot(i+1)));
                         	      
		    end
		    
		end

                if k==kord 
		   if B(i,k-2)~=0 
                      dB2avx(xnr,i)=(k-1)*(k-2)*B(i,k-2)/((tknot(i+k-1)-tknot(i))*(tknot(i+k-2)-tknot(i)));
                   end 
                   if B(i+1,k-2)~=0 
                      dB2avx(xnr,i)=dB2avx(xnr,i)-(k-1)*(k-2)*(...
                                          B(i+1,k-2)/((tknot(i+k-1)-tknot(i))*(tknot(i+k-1)-tknot(i+1)))+...
					  B(i+1,k-2)/((tknot(i+k)-tknot(i+1))*(tknot(i+k-1)-tknot(i+1))));
                   end 
                   if B(i+2,k-2)~=0 
                      dB2avx(xnr,i)=dB2avx(xnr,i)+(k-1)*(k-2)*B(i+2,k-2)/((tknot(i+k)-tknot(i+1))*(tknot(i+k)-tknot(i+2)));
                   end 
                end 
	    
				
	    end
	end
	
    end
    
    
    
    for indexi=istart:islut 
	
	if  (xnr == length(xintervall) & indexi ==islut) 
	    Bavx(xnr,indexi)= 1;
	    
	    dBavx(xnr,indexi) = (kord-1)/(tknot(length(tknot))-tknot(islut));

            dB2avx(xnr,indexi) =(kord-1)*(kord-2)/((tknot(length(tknot))-tknot(islut))*(tknot(length(tknot)-1)-tknot(islut)));
	    	    	    
        elseif (xnr == length(xintervall) & indexi ==islut-1 )
            dBavx(xnr,indexi) = -(kord-1)/(tknot(length(tknot))-tknot(islut));
            dB2avx(xnr,indexi) =-(kord-1)*(kord-2)/((tknot(length(tknot))-tknot(islut))*(tknot(length(tknot)-1)-tknot(islut)))...
                                -(kord-1)*(kord-2)/((tknot(length(tknot))-tknot(islut-1))*(tknot(length(tknot)-1)-tknot(islut)));

        elseif (xnr == length(xintervall) & indexi ==islut-2 )
            dB2avx(xnr,indexi) =(kord-1)*(kord-2)/((tknot(length(tknot))-tknot(islut-1))*(tknot(length(tknot)-1)-tknot(islut)));
	else
	
	    Bavx(xnr,indexi)=B(indexi,kord);
	
	
	end
    
       
	
    end
    
    
    
end

end














