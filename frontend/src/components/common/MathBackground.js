import React, { useEffect, useRef } from 'react';

const MathBackground = () => {
  const containerRef = useRef(null);
  
  useEffect(() => {
    const container = containerRef.current;
    const mathSymbols = [
      '∫', '∑', '∏', '√', 'π', '∞', 'λ', 'θ', 'ω', 'Ω', 'δ', 'Δ', 'α', 'β', 'γ',
      'e^{iπ}+1=0', 'F=ma', 'E=mc^2', 'a^2+b^2=c^2', 'P(A|B)', 'f(x)=ax^2+bx+c',
      '∇×F', '∮', '∂f/∂x', '∫f(x)dx', '∑_{i=1}^{n}', 'lim_{x→∞}'
    ];

    // Créer et ajouter 20 équations flottantes
    for (let i = 0; i < 20; i++) {
      const equation = document.createElement('div');
      const symbol = mathSymbols[Math.floor(Math.random() * mathSymbols.length)];
      
      equation.textContent = symbol;
      equation.style.position = 'absolute';
      equation.style.color = 'rgba(255, 255, 255, 0.05)';
      equation.style.fontSize = `${Math.random() * 40 + 20}px`;
      equation.style.fontFamily = 'Georgia, serif';
      equation.style.left = `${Math.random() * 100}%`;
      equation.style.top = `${Math.random() * 100}%`;
      equation.style.transform = `rotate(${Math.random() * 360}deg)`;
      equation.style.userSelect = 'none';
      equation.style.pointerEvents = 'none';
      
      // Animation
      equation.animate(
        [
          { transform: `translate(0, 0) rotate(${Math.random() * 360}deg)` },
          { transform: `translate(${Math.random() * 40 - 20}px, ${Math.random() * 40 - 20}px) rotate(${Math.random() * 360}deg)` }
        ],
        {
          duration: Math.random() * 15000 + 15000,
          iterations: Infinity,
          direction: 'alternate',
          easing: 'ease-in-out'
        }
      );
      
      container.appendChild(equation);
    }
    
    return () => {
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    };
  }, []);

  return (
    <div 
      ref={containerRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        overflow: 'hidden',
        pointerEvents: 'none'
      }}
    />
  );
};

export default MathBackground;