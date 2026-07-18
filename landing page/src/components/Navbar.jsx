import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X } from 'lucide-react';
import Container from './Common/Container';
import { fadeDown } from '../utils/animations';

export const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { label: 'Features', href: '#features' },
    { label: 'Technology', href: '#technology' },
    { label: 'Solutions', href: '#solutions' },
    { label: 'About', href: '#about' },
    { label: 'Contact', href: '#contact' },
  ];

  return (
    <motion.header 
      initial="hidden"
      animate="visible"
      variants={fadeDown}
      custom={{ delay: 0.15, duration: 0.8 }}
      className="sticky top-0 z-50 w-full pt-4 md:pt-6 transition-all duration-300"
    >
      <Container>
        <div 
          className={`grid grid-cols-2 md:grid-cols-[200px_1fr_200px] items-center px-6 rounded-full border transition-all duration-300 ${
            isScrolled 
              ? 'py-2.5 bg-brand-surface/85 border-brand-border backdrop-blur-2xl shadow-xl shadow-black/10' 
              : 'py-4 bg-brand-surface/30 border-brand-border/40 backdrop-blur-md'
          }`}
        >
          {/* Logo Column */}
          <div className="flex justify-start">
            <a href="#" className="flex items-center gap-2 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent focus-visible:ring-offset-2 focus-visible:ring-offset-brand-bg rounded-lg px-2 py-0.5">
              <span className="font-serif text-[23px] md:text-[27px] font-extrabold tracking-tight text-brand-accent drop-shadow-[0_0_10px_rgba(232,198,138,0.25)] group-hover:text-brand-text transition-all duration-300">
                FarmSense<span className="text-brand-text">.</span>
              </span>
            </a>
          </div>

          {/* Desktop Nav Links Column */}
          <div className="hidden md:flex justify-center items-center gap-8">
            {navLinks.map((link) => (
              <a 
                key={link.label}
                href={link.href}
                className="relative text-sm font-sans font-medium text-brand-secondary hover:text-brand-text transition-colors duration-300 py-1 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent rounded px-1"
              >
                <span>{link.label}</span>
                <span className="absolute bottom-0 left-0 w-full h-[1px] bg-brand-accent scale-x-0 origin-left group-hover:scale-x-100 transition-transform duration-300" />
              </a>
            ))}
          </div>

          {/* Right Column: Get Started on desktop / Mobile Menu button on mobile */}
          <div className="flex justify-end items-center gap-4">
            <div className="hidden md:block">
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => window.location.href = '/dashboard.html'}
                className="px-5 py-2 rounded-full text-xs font-sans font-semibold tracking-wider uppercase bg-brand-accent text-brand-bg hover:opacity-90 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent focus-visible:ring-offset-2 focus-visible:ring-offset-brand-bg"
              >
                Get Started
              </motion.button>
            </div>

            <button 
              onClick={() => setIsOpen(!isOpen)}
              className="p-1 text-brand-secondary hover:text-brand-text md:hidden focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent rounded transition-colors duration-200 cursor-pointer"
              aria-label={isOpen ? "Close main menu" : "Open main menu"}
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown Menu */}
        <AnimatePresence>
          {isOpen && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="md:hidden mt-2 p-6 rounded-2xl border border-brand-border bg-brand-surface/95 backdrop-blur-xl shadow-xl"
            >
              <div className="flex flex-col gap-4">
                {navLinks.map((link) => (
                  <a 
                    key={link.label}
                    href={link.href}
                    onClick={() => setIsOpen(false)}
                    className="text-base font-sans font-medium text-brand-secondary hover:text-brand-accent py-2 border-b border-brand-border/40 transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent rounded"
                  >
                    {link.label}
                  </a>
                ))}
                <button 
                  onClick={() => {
                    setIsOpen(false);
                    window.location.href = '/dashboard.html';
                  }}
                  className="w-full mt-4 py-3 rounded-full text-sm font-sans font-semibold tracking-wider uppercase border border-brand-accent bg-brand-accent text-brand-bg hover:opacity-95 transition-all duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent"
                >
                  Get Started
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Container>
    </motion.header>
  );
};

export default Navbar;
