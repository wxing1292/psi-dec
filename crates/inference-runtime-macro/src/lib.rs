use proc_macro::TokenStream;
use quote::quote;
use syn::Expr;
use syn::ItemFn;
use syn::parse_macro_input;

#[proc_macro_attribute]
pub fn sanity_check(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut check_expr: Option<Expr> = None;

    let parser = syn::meta::parser(|meta: syn::meta::ParseNestedMeta| {
        if meta.path.is_ident("sanity_check_fn") {
            let lit: syn::LitStr = meta.value()?.parse()?;
            check_expr = Some(lit.parse()?);
            Ok(())
        } else {
            Err(meta.error("unknown attribute key (expected `sanity_check_fn`)"))
        }
    });

    parse_macro_input!(attr with parser);

    let check_expr = match check_expr {
        Some(e) => e,
        None => {
            return quote! {
                compile_error!("missing required: sanity_check_fn = \"self.sanity_check()\"");
            }
            .into();
        },
    };

    let item_fn = parse_macro_input!(item as ItemFn);
    let vis = &item_fn.vis;
    let sig = &item_fn.sig;
    let attrs = &item_fn.attrs;
    let block = &item_fn.block;

    quote! {
        #(#attrs)*
        #vis #sig {
            #[cfg(debug_assertions)]
            { #check_expr; }

            let ret = 'sanity_ret: { #block };

            #[cfg(debug_assertions)]
            { #check_expr; }

            ret
        }
    }
    .into()
}
