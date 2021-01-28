tic
%Limpar Variáveis
clear
clc

%Variáveis para a Fase de Treinamento
Dados=dlmread('timp3.sa.txt');
%normalização dos dados
AB=max(Dados);
AC=min(Dados);
ma=max(Dados(:,1));
mi=min(Dados(:,1));
[l,c]=size(Dados);
index=randperm(l);
%divisão em dois grupos
index_1=index(201:247); %teste
index_2=index(1:200); %treino

for i=1:l
    Dados(i,:)=(Dados(i,:)-AC)./(AB-AC);
end

x1=Dados(index_2,2:6);
[b,b1]=size(x1);
bias=-ones(b,1);
x=[bias'; x1'];
[n,m]=size(x);
%Inserindo a saída desejada d
d=Dados(index_2,1);
nne=5;         %número de neurônios de entrada
nnc=2;         %número de camadas ocultas
nno=5;        %número de neurônios da camada oculta
nns=1;         %número de neurônios da camada de saida

%Variáveis para a Fase de teste
xx1=Dados(index_1,2:6);
d1=Dados(index_1,1);
[bb,bb1]=size(xx1);

biass=-ones(bb,1);
xx=[biass'; xx1'];
eta=0.5;        %Taxa de aprendizagem
ep=5e-8;        %Precisão - Epsilon
g=inline('1./(1+exp(-2*u))');                  %função de ativação
g1=inline('2*exp(-2*u)./((1+exp(-2*u)).^2)');   %derivada da função de ativação
%Fase de Treinamento

    w1=rand(nno,nne+1);
    w2=rand(nns,nno+1);
    epoca=0;
    eqm=10000;        %Erro quadratico médio
    e=1;
    t=1;
    while(e>=ep)&&(epoca<1000)
        eqmanterior=eqm;
        eqm=0;
        for i=1:m
            i1=w1*x(:,i);          %entrada da camada intermediaria
            y1=g(i1);              %saida do neuronio da camada intermediaria
            y11=[-1 y1']';         %saida com bias
            i2=w2*y11;             %entrada da camada de saida
            y2(i)=g(i2);           %saida da rede
            ee=d(i)-y2(i);
            delta2=ee*g1(i2);
            w2=w2+eta*delta2*y11';
            for j=1:nno
                delta1(j)=0;
                delta1(j)=delta2*w2(j+1)*g1(i1(j));
            end
            w1=w1+eta*delta1'*x(:,i)';
        end
        for j=1:m
            i1=w1*x(:,j);
            y1=g(i1);
            y11=[-1 y1']';
            i2=w2*y11;
            y2(j)=g(i2);
            ee=d(j)-y2(j);
            eqm=eqm+ee^2;
            
        end
       eqm=eqm/m;
       epoca=epoca+1;
       eqmt(epoca)=eqm;
       e=abs(eqm-eqmanterior)
       erro(epoca)=e;
       eqmanterior=eqm;
    end
    
    for i=1:bb
        I1=w1*xx(:,i);
        Y1=g(I1);
        Y11=[-1 Y1'];
        I2=w2*Y11';
        Y2(i)=g(I2);
    end
    
    cont=0;
    
    for i=1:length(Y2)
    %Y2(i)=Y2(i)*(AB(7)-AC(7))+AC(7);
    %d1(i)=d1(i)*(AB(7)-AC(7))+AC(7);
    ErrM(i)=abs((d1(i)-Y2(i))/d1(i));
    end
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%% ERROS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Eqm_rna=mean(100*abs(Y2'-d1)./d1);
   
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  amostra=[1:length(index_1)];
%   figure(1)
%     plot(amostra,d1,'--*r',amostra,Y2,'--*b')
%     title ('Fase de teste');
%     xlabel('Amostras')
%     ylabel('Prazos')
%     legend('Valor medido','Saída da rede')
%        
    %%%%%%%%%%%%%%%%%%%%%%% Correlações%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    rho_rna = corr(d1,Y2');
   
   %%
   d1_desnormalizado = (d1-min(d1))*(ma-mi)/(max(d1)-min(d1))+mi;
   y2_desnormalizado = (Y2-min(Y2))*(ma-mi)/(max(Y2)-min(Y2))+mi;
   
   figure(1)
    plot(amostra,d1_desnormalizado,'--*r',amostra,y2_desnormalizado,'--*b')
    title ('Fase de teste');
    xlabel('Amostras')
    ylabel('Valores')
    legend('Valor medido','Saída da rede')
   